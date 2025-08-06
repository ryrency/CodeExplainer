"""
Configuration management for the MCP Code Q&A system.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for the MCP Code Q&A system."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Model settings
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4-turbo-preview"
    
    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_results: int = 5
    
    # Vector store settings
    vector_store_path: str = "data/vector_store"
    collection_name: str = "code_chunks"
    
    # File processing settings
    supported_extensions: tuple = (".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp")
    max_file_size_mb: int = 10
    
    # Evaluation settings
    evaluation_data_path: str = "data/evaluation"
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Override with environment variables if present
        if os.getenv("EMBEDDING_MODEL"):
            self.embedding_model = os.getenv("EMBEDDING_MODEL")
        
        if os.getenv("LLM_MODEL"):
            self.llm_model = os.getenv("LLM_MODEL")
        
        if os.getenv("CHUNK_SIZE"):
            self.chunk_size = int(os.getenv("CHUNK_SIZE"))
        
        if os.getenv("CHUNK_OVERLAP"):
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))
        
        if os.getenv("MAX_RETRIEVAL_RESULTS"):
            self.max_retrieval_results = int(os.getenv("MAX_RETRIEVAL_RESULTS"))
        
        if os.getenv("VECTOR_STORE_PATH"):
            self.vector_store_path = os.getenv("VECTOR_STORE_PATH")
        
        # Create necessary directories
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
        Path(self.evaluation_data_path).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.openai_api_key and not self.anthropic_api_key:
            raise ValueError("Either OPENAI_API_KEY or ANTHROPIC_API_KEY must be set")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        return True 