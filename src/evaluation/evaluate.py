"""
Main evaluation script for the RAG system.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

from ..rag_system.rag_manager import RAGManager
from ..utils.config import Config
from .evaluator import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate RAG system quality')
    parser.add_argument('--repository', '-r', required=True, 
                       help='Path to the repository to evaluate')
    parser.add_argument('--dataset', '-d', 
                       help='Path to the evaluation dataset (JSON format)')
    parser.add_argument('--output', '-o', default='data/evaluation/results.json',
                       help='Output path for evaluation results')
    parser.add_argument('--report', default='data/evaluation/report.txt',
                       help='Output path for evaluation report')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample dataset for evaluation')
    parser.add_argument('--force-reindex', action='store_true',
                       help='Force reindexing of the repository')
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = Config()
        config.validate()
        
        # Initialize RAG manager
        rag_manager = RAGManager(config)
        
        # Check if repository exists
        repo_path = Path(args.repository)
        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {args.repository}")
            sys.exit(1)
        
        # Index repository if needed
        if not rag_manager.is_repository_indexed(args.repository) or args.force_reindex:
            logger.info("Indexing repository...")
            stats = await rag_manager.index_repository(args.repository, args.force_reindex)
            logger.info(f"Indexing completed: {stats['chunks_created']} chunks created")
        else:
            logger.info("Repository already indexed")
        
        # Initialize evaluator
        evaluator = RAGEvaluator(config, rag_manager)
        
        # Load evaluation dataset
        if args.sample:
            logger.info("Using sample dataset")
            qa_pairs = evaluator.create_sample_dataset()
        elif args.dataset:
            logger.info(f"Loading dataset from {args.dataset}")
            qa_pairs = await evaluator.load_grip_qa_dataset(args.dataset)
            if not qa_pairs:
                logger.error("No Q&A pairs found in dataset")
                sys.exit(1)
        else:
            logger.error("Please specify either --dataset or --sample")
            sys.exit(1)
        
        # Run evaluation
        logger.info(f"Starting evaluation with {len(qa_pairs)} Q&A pairs")
        summary = await evaluator.evaluate_dataset(qa_pairs, args.repository)
        
        # Save results
        evaluator.save_evaluation_results(summary, args.output)
        
        # Generate and save report
        report = evaluator.generate_evaluation_report(summary)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions: {summary.total_questions}")
        print(f"Average ROUGE F1: {summary.average_rouge_f1:.4f}")
        print(f"Average BLEU: {summary.average_bleu:.4f}")
        print(f"Average Semantic Similarity: {summary.average_semantic_similarity:.4f}")
        print(f"Average Overall Score: {summary.average_overall_score:.4f}")
        print(f"Evaluation Time: {summary.evaluation_time:.2f} seconds")
        print(f"\nResults saved to: {args.output}")
        print(f"Report saved to: {args.report}")
        
        # Performance assessment
        if summary.average_overall_score >= 0.8:
            print("\n🎉 EXCELLENT performance!")
        elif summary.average_overall_score >= 0.6:
            print("\n✅ GOOD performance")
        elif summary.average_overall_score >= 0.4:
            print("\n⚠️  FAIR performance - consider improvements")
        else:
            print("\n❌ POOR performance - significant improvements needed")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 