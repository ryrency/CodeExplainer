"""
RAG Manager for coordinating code chunking, indexing, retrieval, and generation.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..code_parser.parser import CodeParser, CodeChunk
from ..utils.config import Config

logger = logging.getLogger(__name__)

class RAGManager:
    """Manages the RAG system for code Q&A."""
    
    def __init__(self, config: Config):
        self.config = config
        self.code_parser = CodeParser()
        
        # Initialize vector store
        self.vector_store_path = Path(config.vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_store_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embeddings
        if config.openai_api_key:
            self.embeddings = OpenAIEmbeddings(
                model=config.embedding_model,
                openai_api_key=config.openai_api_key
            )
        else:
            raise ValueError("OpenAI API key is required for embeddings")
        
        # Initialize LLM
        if config.openai_api_key:
            self.llm = ChatOpenAI(
                model=config.llm_model,
                openai_api_key=config.openai_api_key,
                temperature=0.1
            )
        else:
            raise ValueError("OpenAI API key is required for LLM")
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=config.collection_name,
            embedding_function=self.embeddings
        )
        
        # Repository metadata storage
        self.repo_metadata_file = self.vector_store_path / "repository_metadata.json"
        self.repo_metadata = self._load_repo_metadata()
    
    def _load_repo_metadata(self) -> Dict[str, Any]:
        """Load repository metadata from file."""
        if self.repo_metadata_file.exists():
            try:
                with open(self.repo_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading repository metadata: {e}")
                return {}
        return {}
    
    def _save_repo_metadata(self):
        """Save repository metadata to file."""
        try:
            with open(self.repo_metadata_file, 'w') as f:
                json.dump(self.repo_metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving repository metadata: {e}")
    
    async def index_repository(self, repository_path: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Index a repository for Q&A."""
        repo_path = Path(repository_path).resolve()
        repo_key = str(repo_path)
        
        # Check if already indexed and not forcing reindex
        if not force_reindex and repo_key in self.repo_metadata:
            logger.info(f"Repository {repository_path} already indexed")
            return self.repo_metadata[repo_key]
        
        logger.info(f"Indexing repository: {repository_path}")
        start_time = time.time()
        
        try:
            # Parse code into chunks
            chunks = self.code_parser.parse_directory(repo_path)
            logger.info(f"Parsed {len(chunks)} code chunks from {repository_path}")
            
            if not chunks:
                logger.warning(f"No code chunks found in {repository_path}")
                return {
                    'files_processed': 0,
                    'chunks_created': 0,
                    'indexing_time': time.time() - start_time
                }
            
            # Convert chunks to documents for vector store
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Debug logging
                    logger.info(f"Processing chunk {i}: type={type(chunk)}, content={str(chunk)[:100]}...")
                    
                    # Check if chunk is a string instead of CodeChunk object
                    if isinstance(chunk, str):
                        logger.warning(f"Chunk {i} is a string, skipping: {chunk[:100]}...")
                        continue
                    
                    # Validate chunk has required attributes
                    if not hasattr(chunk, 'file_path') or not hasattr(chunk, 'chunk_type'):
                        logger.warning(f"Chunk {i} missing required attributes, skipping")
                        continue
                    
                    # Check if chunk content is empty
                    if not hasattr(chunk, 'content') or not chunk.content or not chunk.content.strip():
                        logger.warning(f"Chunk {i} has empty content, skipping")
                        continue
                    
                    # Create document content with metadata
                    doc_content = f"File: {chunk.file_path}\n"
                    doc_content += f"Type: {chunk.chunk_type}\n"
                    doc_content += f"Lines: {chunk.start_line}-{chunk.end_line}\n"
                    doc_content += f"Content:\n{chunk.content}"
                    
                    # Debug: log document content length
                    logger.info(f"Document content length for chunk {i}: {len(doc_content)}")
                    
                    documents.append(doc_content)
                    
                    # Create metadata
                    metadata = {
                        'repository_path': repo_key,
                        'file_path': chunk.file_path,
                        'chunk_type': chunk.chunk_type,
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line,
                    }
                    
                    # Add chunk metadata if it exists and is a dict
                    if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                        metadata.update(chunk.metadata)
                    
                    # Ensure all metadata values are simple types that ChromaDB accepts
                    filtered_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, (str, bool, int, float)):
                            filtered_metadata[key] = value
                        else:
                            filtered_metadata[key] = str(value)
                    
                    metadatas.append(filtered_metadata)
                    
                    # Create unique ID
                    chunk_id = f"{repo_key}_{i}_{chunk.chunk_type}_{chunk.start_line}"
                    ids.append(chunk_id)
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    continue
            
            # Remove existing documents for this repository if reindexing
            if force_reindex and repo_key in self.repo_metadata:
                await self._remove_repository_documents(repo_key)
            
            # Filter out empty documents before adding to vector store
            valid_documents = []
            valid_metadatas = []
            valid_ids = []
            
            for doc, meta, doc_id in zip(documents, metadatas, ids):
                if doc and doc.strip():  # Only add non-empty documents
                    valid_documents.append(Document(page_content=doc, metadata=meta))
                    valid_metadatas.append(meta)
                    valid_ids.append(doc_id)
                else:
                    logger.warning(f"Skipping empty document with ID: {doc_id}")
            
            if not valid_documents:
                logger.error("No valid documents to add to vector store")
                raise ValueError("No valid documents found for indexing")
            
            # Add documents to vector store
            self.vector_store.add_documents(
                documents=valid_documents,
                ids=valid_ids
            )
            
            # Update repository metadata
            indexing_time = time.time() - start_time
            self.repo_metadata[repo_key] = {
                'repository_path': repo_key,
                'files_processed': len(set(meta['file_path'] for meta in valid_metadatas)),
                'chunks_created': len(valid_documents),
                'indexing_time': indexing_time,
                'last_indexed': datetime.now().isoformat(),
                'chunk_types': list(set(meta['chunk_type'] for meta in valid_metadatas)),
                'file_types': list(set(Path(meta['file_path']).suffix for meta in valid_metadatas))
            }
            
            self._save_repo_metadata()
            
            logger.info(f"Successfully indexed {repository_path} in {indexing_time:.2f} seconds")
            return self.repo_metadata[repo_key]
            
        except Exception as e:
            logger.error(f"Error indexing repository {repository_path}: {e}")
            raise
    
    async def _remove_repository_documents(self, repository_path: str):
        """Remove all documents for a specific repository."""
        try:
            # Get all documents for this repository
            results = self.vector_store.get(
                where={"repository_path": repository_path}
            )
            
            if results['ids']:
                # Remove documents by IDs
                self.vector_store.delete(ids=results['ids'])
                logger.info(f"Removed {len(results['ids'])} documents for repository {repository_path}")
        
        except Exception as e:
            logger.error(f"Error removing repository documents: {e}")
    
    async def ask_question(self, question: str, repository_path: str) -> str:
        """Ask a question about code in the indexed repository."""
        repo_path = Path(repository_path).resolve()
        repo_key = str(repo_path)
        
        if repo_key not in self.repo_metadata:
            raise ValueError(f"Repository {repository_path} is not indexed")
        
        try:
            # Retrieve relevant code snippets
            snippets = await self.get_relevant_snippets(question, repository_path, self.config.max_retrieval_results)
            
            if not snippets:
                return "I couldn't find any relevant code snippets to answer your question. Please try rephrasing your question or ensure the repository is properly indexed."
            
            # Create context from snippets
            context = self._create_context_from_snippets(snippets)
            
            # Generate answer using LLM
            prompt = self._create_qa_prompt(question, context)
            
            response = await self.llm.ainvoke(prompt)
            answer = response.content
            
            return answer
            
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return f"Error processing your question: {str(e)}"
    
    async def get_relevant_snippets(self, query: str, repository_path: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Get relevant code snippets for a query."""
        repo_path = Path(repository_path).resolve()
        repo_key = str(repo_path)
        
        try:
            # Search in vector store
            results = self.vector_store.similarity_search_with_score(
                query,
                k=max_results * 2,  # Get more results to filter by repository
                filter={"repository_path": repo_key}
            )
            
            snippets = []
            for doc, score in results:
                snippet = {
                    'code': doc.page_content,
                    'file_path': doc.metadata.get('file_path', ''),
                    'chunk_type': doc.metadata.get('chunk_type', ''),
                    'start_line': doc.metadata.get('start_line', 0),
                    'end_line': doc.metadata.get('end_line', 0),
                    'score': float(score),
                    'metadata': doc.metadata
                }
                snippets.append(snippet)
            
            # Sort by score and return top results
            snippets.sort(key=lambda x: x['score'])
            return snippets[:max_results]
            
        except Exception as e:
            logger.error(f"Error retrieving snippets: {e}")
            return []
    
    def _create_context_from_snippets(self, snippets: List[Dict[str, Any]]) -> str:
        """Create context string from code snippets."""
        context_parts = []
        
        for i, snippet in enumerate(snippets, 1):
            context_parts.append(f"Code Snippet {i}:")
            context_parts.append(f"File: {snippet['file_path']}")
            context_parts.append(f"Type: {snippet['chunk_type']}")
            context_parts.append(f"Lines: {snippet['start_line']}-{snippet['end_line']}")
            context_parts.append(f"Relevance Score: {snippet['score']:.3f}")
            context_parts.append("Code:")
            context_parts.append(snippet['code'])
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _create_qa_prompt(self, question: str, context: str) -> str:
        """Create prompt for Q&A generation."""
        return f"""You are an expert code analyst. Answer the following question about the code based on the provided context.

Question: {question}

Context (relevant code snippets):
{context}

Instructions:
1. Answer the question based on the code context provided
2. If the context doesn't contain enough information to answer the question, say so
3. Be specific and reference the relevant code snippets when possible
4. If you need to explain code, do so clearly and concisely
5. If the question is about implementation details, focus on the actual code shown

Answer:"""
    
    def is_repository_indexed(self, repository_path: str) -> bool:
        """Check if a repository is indexed."""
        repo_path = Path(repository_path).resolve()
        repo_key = str(repo_path)
        return repo_key in self.repo_metadata
    
    def list_indexed_repositories(self) -> List[Dict[str, Any]]:
        """List all indexed repositories."""
        repositories = []
        
        for repo_key, metadata in self.repo_metadata.items():
            # Count chunks for this repository
            try:
                results = self.vector_store.get(
                    where={"repository_path": repo_key}
                )
                chunk_count = len(results['ids']) if results['ids'] else 0
            except:
                chunk_count = metadata.get('chunks_created', 0)
            
            repositories.append({
                'path': repo_key,
                'chunk_count': chunk_count,
                'last_indexed': metadata.get('last_indexed', 'Unknown'),
                'files_processed': metadata.get('files_processed', 0)
            })
        
        return repositories
    
    def get_repository_stats(self, repository_path: str) -> Optional[Dict[str, Any]]:
        """Get statistics about an indexed repository."""
        repo_path = Path(repository_path).resolve()
        repo_key = str(repo_path)
        
        if repo_key not in self.repo_metadata:
            return None
        
        metadata = self.repo_metadata[repo_key]
        
        # Get additional stats from vector store
        try:
            results = self.vector_store.get(
                where={"repository_path": repo_key}
            )
            total_chunks = len(results['ids']) if results['ids'] else 0
        except:
            total_chunks = metadata.get('chunks_created', 0)
        
        # Calculate index size
        index_size_mb = 0
        try:
            if self.vector_store_path.exists():
                index_size_mb = sum(f.stat().st_size for f in self.vector_store_path.rglob('*') if f.is_file()) / (1024 * 1024)
        except:
            pass
        
        return {
            'total_files': metadata.get('files_processed', 0),
            'total_chunks': total_chunks,
            'file_types': metadata.get('file_types', []),
            'chunk_types': metadata.get('chunk_types', []),
            'index_size_mb': index_size_mb,
            'last_indexed': metadata.get('last_indexed', 'Unknown'),
            'indexing_time': metadata.get('indexing_time', 0)
        }
    
    async def delete_repository(self, repository_path: str) -> bool:
        """Delete a repository from the index."""
        repo_path = Path(repository_path).resolve()
        repo_key = str(repo_path)
        
        if repo_key not in self.repo_metadata:
            return False
        
        try:
            # Remove documents from vector store
            await self._remove_repository_documents(repo_key)
            
            # Remove from metadata
            del self.repo_metadata[repo_key]
            self._save_repo_metadata()
            
            logger.info(f"Successfully deleted repository {repository_path} from index")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting repository {repository_path}: {e}")
            return False 