#!/usr/bin/env python3
"""
FastAPI wrapper for the MCP Code Q&A Server
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add src to path and set up imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import with proper module paths
from src.utils.config import Config
from src.rag_system.rag_manager import RAGManager
from src.code_parser.parser import CodeParser

app = FastAPI(
    title="MCP Code Q&A API",
    description="REST API wrapper for the MCP Code Q&A Server",
    version="1.0.0"
)

# Initialize components
config = Config()
rag_manager = RAGManager(config)
code_parser = CodeParser()

# Pydantic models for API requests/responses
class IndexRepositoryRequest(BaseModel):
    repository_path: str
    force_reindex: bool = False

class AskQuestionRequest(BaseModel):
    question: str
    repository_path: str

class GetContextRequest(BaseModel):
    query: str
    repository_path: str
    max_results: int = 5

class IndexRepositoryResponse(BaseModel):
    success: bool
    message: str
    chunk_count: Optional[int] = None

class AskQuestionResponse(BaseModel):
    answer: str
    success: bool

class GetContextResponse(BaseModel):
    snippets: list
    success: bool

class ListRepositoriesResponse(BaseModel):
    repositories: list
    success: bool

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MCP Code Q&A API",
        "version": "1.0.0",
        "endpoints": [
            "/index-repository",
            "/ask-question", 
            "/get-context",
            "/list-repositories",
            "/repository-stats"
        ]
    }

@app.post("/index-repository", response_model=IndexRepositoryResponse)
async def index_repository(request: IndexRepositoryRequest):
    """Index a repository for Q&A."""
    try:
        result = await rag_manager.index_repository(
            request.repository_path, 
            force_reindex=request.force_reindex
        )
        
        # Get stats for response
        stats = rag_manager.get_repository_stats(request.repository_path)
        chunk_count = stats.get('total_chunks', 0) if stats else 0
        
        return IndexRepositoryResponse(
            success=True,
            message=str(result),
            chunk_count=chunk_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question", response_model=AskQuestionResponse)
async def ask_question(request: AskQuestionRequest):
    """Ask a question about code in a repository."""
    try:
        answer = await rag_manager.ask_question(
            request.question,
            request.repository_path
        )
        
        return AskQuestionResponse(
            answer=answer,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-context", response_model=GetContextResponse)
async def get_context(request: GetContextRequest):
    """Get relevant code snippets for a query."""
    try:
        snippets = await rag_manager.get_relevant_snippets(
            request.query,
            request.repository_path,
            max_results=request.max_results
        )
        
        return GetContextResponse(
            snippets=snippets,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-repositories", response_model=ListRepositoriesResponse)
async def list_repositories():
    """List all indexed repositories."""
    try:
        repositories = rag_manager.list_indexed_repositories()
        
        return ListRepositoriesResponse(
            repositories=repositories,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repository-stats/{repository_path:path}")
async def get_repository_stats(repository_path: str):
    """Get statistics for a specific repository."""
    try:
        stats = rag_manager.get_repository_stats(repository_path)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Repository not found or not indexed")
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_key_configured": bool(config.openai_api_key),
        "embedding_model": config.embedding_model,
        "llm_model": config.llm_model
    }

if __name__ == "__main__":
    print("🚀 Starting MCP Code Q&A API Server...")
    print("📖 API Documentation available at: http://localhost:8000/docs")
    print("🔗 Health check: http://localhost:8000/health")
    
    uvicorn.run(
        "api_wrapper:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 