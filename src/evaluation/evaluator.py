"""
Evaluation framework for measuring RAG system quality.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from ..rag_system.rag_manager import RAGManager
from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Results from evaluating a single Q&A pair."""
    question: str
    reference_answer: str
    generated_answer: str
    rouge_scores: Dict[str, float]
    bleu_score: float
    semantic_similarity: float
    retrieval_accuracy: float
    answer_length_ratio: float
    overall_score: float

@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""
    total_questions: int
    average_rouge_f1: float
    average_bleu: float
    average_semantic_similarity: float
    average_retrieval_accuracy: float
    average_overall_score: float
    detailed_results: List[EvaluationResult]
    evaluation_time: float

class RAGEvaluator:
    """Evaluates RAG system quality using multiple metrics."""
    
    def __init__(self, config: Config, rag_manager: RAGManager):
        self.config = config
        self.rag_manager = rag_manager
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
        # Initialize embeddings for semantic similarity
        if config.openai_api_key:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                model=config.embedding_model,
                openai_api_key=config.openai_api_key
            )
        else:
            self.embeddings = None
    
    async def evaluate_qa_pair(self, question: str, reference_answer: str, repository_path: str) -> EvaluationResult:
        """Evaluate a single Q&A pair."""
        try:
            # Generate answer using RAG system
            generated_answer = await self.rag_manager.ask_question(question, repository_path)
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(reference_answer, generated_answer)
            rouge_f1 = (rouge_scores['rouge1'].fmeasure + 
                       rouge_scores['rouge2'].fmeasure + 
                       rouge_scores['rougeL'].fmeasure) / 3
            
            # Calculate BLEU score
            reference_tokens = reference_answer.split()
            generated_tokens = generated_answer.split()
            bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=self.smoothing)
            
            # Calculate semantic similarity
            semantic_similarity = await self._calculate_semantic_similarity(reference_answer, generated_answer)
            
            # Calculate retrieval accuracy (simplified)
            retrieval_accuracy = self._calculate_retrieval_accuracy(question, generated_answer, reference_answer)
            
            # Calculate answer length ratio
            length_ratio = len(generated_answer.split()) / max(len(reference_answer.split()), 1)
            length_ratio = min(length_ratio, 2.0) / 2.0  # Normalize to 0-1
            
            # Calculate overall score (weighted average)
            overall_score = (
                0.3 * rouge_f1 +
                0.2 * bleu_score +
                0.3 * semantic_similarity +
                0.1 * retrieval_accuracy +
                0.1 * length_ratio
            )
            
            return EvaluationResult(
                question=question,
                reference_answer=reference_answer,
                generated_answer=generated_answer,
                rouge_scores={
                    'rouge1_f1': rouge_scores['rouge1'].fmeasure,
                    'rouge2_f1': rouge_scores['rouge2'].fmeasure,
                    'rougeL_f1': rouge_scores['rougeL'].fmeasure,
                    'average_f1': rouge_f1
                },
                bleu_score=bleu_score,
                semantic_similarity=semantic_similarity,
                retrieval_accuracy=retrieval_accuracy,
                answer_length_ratio=length_ratio,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Error evaluating Q&A pair: {e}")
            # Return default result for failed evaluation
            return EvaluationResult(
                question=question,
                reference_answer=reference_answer,
                generated_answer="Error generating answer",
                rouge_scores={'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0, 'average_f1': 0.0},
                bleu_score=0.0,
                semantic_similarity=0.0,
                retrieval_accuracy=0.0,
                answer_length_ratio=0.0,
                overall_score=0.0
            )
    
    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings."""
        if not self.embeddings:
            return 0.0
        
        try:
            # Get embeddings
            embedding1 = await self.embeddings.aembed_query(text1)
            embedding2 = await self.embeddings.aembed_query(text2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_retrieval_accuracy(self, question: str, generated_answer: str, reference_answer: str) -> float:
        """Calculate retrieval accuracy based on answer relevance."""
        # Simple heuristic: check if key terms from question appear in answer
        question_terms = set(question.lower().split())
        answer_terms = set(generated_answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'why', 'when', 'where', 'which', 'who', 'that', 'this', 'these', 'those'}
        question_terms = question_terms - stop_words
        answer_terms = answer_terms - stop_words
        
        if not question_terms:
            return 0.5  # Default score if no meaningful terms
        
        # Calculate overlap
        overlap = len(question_terms.intersection(answer_terms))
        accuracy = overlap / len(question_terms)
        
        return min(1.0, accuracy)
    
    async def evaluate_dataset(self, 
                              qa_pairs: List[Dict[str, str]], 
                              repository_path: str) -> EvaluationSummary:
        """Evaluate a dataset of Q&A pairs."""
        logger.info(f"Starting evaluation of {len(qa_pairs)} Q&A pairs")
        start_time = time.time()
        
        results = []
        for i, qa_pair in enumerate(qa_pairs):
            logger.info(f"Evaluating pair {i+1}/{len(qa_pairs)}")
            result = await self.evaluate_qa_pair(
                qa_pair['question'],
                qa_pair['reference_answer'],
                repository_path
            )
            results.append(result)
        
        # Calculate summary statistics
        evaluation_time = time.time() - start_time
        
        summary = EvaluationSummary(
            total_questions=len(qa_pairs),
            average_rouge_f1=np.mean([r.rouge_scores['average_f1'] for r in results]),
            average_bleu=np.mean([r.bleu_score for r in results]),
            average_semantic_similarity=np.mean([r.semantic_similarity for r in results]),
            average_retrieval_accuracy=np.mean([r.retrieval_accuracy for r in results]),
            average_overall_score=np.mean([r.overall_score for r in results]),
            detailed_results=results,
            evaluation_time=evaluation_time
        )
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        return summary
    
    def save_evaluation_results(self, summary: EvaluationSummary, output_path: str):
        """Save evaluation results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        results_data = {
            'summary': {
                'total_questions': summary.total_questions,
                'average_rouge_f1': summary.average_rouge_f1,
                'average_bleu': summary.average_bleu,
                'average_semantic_similarity': summary.average_semantic_similarity,
                'average_retrieval_accuracy': summary.average_retrieval_accuracy,
                'average_overall_score': summary.average_overall_score,
                'evaluation_time': summary.evaluation_time
            },
            'detailed_results': [
                {
                    'question': r.question,
                    'reference_answer': r.reference_answer,
                    'generated_answer': r.generated_answer,
                    'rouge_scores': r.rouge_scores,
                    'bleu_score': r.bleu_score,
                    'semantic_similarity': r.semantic_similarity,
                    'retrieval_accuracy': r.retrieval_accuracy,
                    'answer_length_ratio': r.answer_length_ratio,
                    'overall_score': r.overall_score
                }
                for r in summary.detailed_results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
    
    def generate_evaluation_report(self, summary: EvaluationSummary) -> str:
        """Generate a human-readable evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append(f"Total Questions Evaluated: {summary.total_questions}")
        report.append(f"Average ROUGE F1 Score: {summary.average_rouge_f1:.4f}")
        report.append(f"Average BLEU Score: {summary.average_bleu:.4f}")
        report.append(f"Average Semantic Similarity: {summary.average_semantic_similarity:.4f}")
        report.append(f"Average Retrieval Accuracy: {summary.average_retrieval_accuracy:.4f}")
        report.append(f"Average Overall Score: {summary.average_overall_score:.4f}")
        report.append(f"Total Evaluation Time: {summary.evaluation_time:.2f} seconds")
        report.append("")
        
        # Performance analysis
        report.append("PERFORMANCE ANALYSIS:")
        if summary.average_overall_score >= 0.8:
            report.append("Overall Performance: EXCELLENT")
        elif summary.average_overall_score >= 0.6:
            report.append("Overall Performance: GOOD")
        elif summary.average_overall_score >= 0.4:
            report.append("Overall Performance: FAIR")
        else:
            report.append("Overall Performance: POOR")
        
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 60)
        
        for i, result in enumerate(summary.detailed_results, 1):
            report.append(f"Question {i}: {result.question}")
            report.append(f"Reference Answer: {result.reference_answer}")
            report.append(f"Generated Answer: {result.generated_answer}")
            report.append(f"ROUGE F1: {result.rouge_scores['average_f1']:.4f}")
            report.append(f"BLEU: {result.bleu_score:.4f}")
            report.append(f"Semantic Similarity: {result.semantic_similarity:.4f}")
            report.append(f"Overall Score: {result.overall_score:.4f}")
            report.append("-" * 60)
        
        return "\n".join(report)
    
    async def load_grip_qa_dataset(self, dataset_path: str) -> List[Dict[str, str]]:
        """Load the GRIP QA dataset."""
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            logger.warning(f"Dataset file not found: {dataset_path}")
            return []
        
        try:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            qa_pairs = []
            for item in data:
                if 'question' in item and 'answer' in item:
                    qa_pairs.append({
                        'question': item['question'],
                        'reference_answer': item['answer']
                    })
            
            logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from dataset")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
    def create_sample_dataset(self) -> List[Dict[str, str]]:
        """Create a sample dataset for testing."""
        return [
            {
                'question': 'What does the main function do?',
                'reference_answer': 'The main function initializes the application and starts the server.'
            },
            {
                'question': 'How is the database connection handled?',
                'reference_answer': 'Database connections are managed through a connection pool with automatic retry logic.'
            },
            {
                'question': 'What are the main classes in this project?',
                'reference_answer': 'The main classes include Config, RAGManager, CodeParser, and Evaluator.'
            },
            {
                'question': 'How does error handling work?',
                'reference_answer': 'Errors are caught and logged with appropriate error messages returned to the user.'
            },
            {
                'question': 'What dependencies does this project use?',
                'reference_answer': 'The project uses OpenAI API, ChromaDB, LangChain, and various Python libraries.'
            }
        ] 