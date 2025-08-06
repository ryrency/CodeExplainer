#!/usr/bin/env python3
"""
Simple test for the RAG system components
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.config import Config
from rag_system.rag_manager import RAGManager
from code_parser.parser import CodeParser

def test_rag_system():
    """Test the RAG system with a simple example."""
    
    print("🧪 Testing RAG System Components...")
    
    # Load configuration
    config = Config()
    print(f"✅ Configuration loaded")
    print(f"   - OpenAI API Key: {'✅ Set' if config.openai_api_key else '❌ Not set'}")
    print(f"   - Embedding Model: {config.embedding_model}")
    print(f"   - LLM Model: {config.llm_model}")
    
    # Initialize components
    rag_manager = RAGManager(config)
    code_parser = CodeParser()
    print("✅ Components initialized")
    
    # Test with current directory
    test_path = Path.cwd()
    print(f"\n📁 Testing with repository: {test_path}")
    
    # Test 1: Parse code
    print("\n1️⃣ Testing code parsing...")
    try:
        chunks = code_parser.parse_repository(test_path)
        print(f"✅ Parsed {len(chunks)} code chunks")
        
        # Show some examples
        for i, chunk in enumerate(chunks[:3]):
            print(f"   Chunk {i+1}: {chunk.file_path} - {chunk.chunk_type}")
    except Exception as e:
        print(f"❌ Error parsing code: {e}")
    
    # Test 2: Index repository
    print("\n2️⃣ Testing repository indexing...")
    try:
        result = rag_manager.index_repository(test_path, force_reindex=True)
        print(f"✅ Indexing result: {result}")
    except Exception as e:
        print(f"❌ Error indexing: {e}")
    
    # Test 3: Ask a question
    print("\n3️⃣ Testing question answering...")
    try:
        answer = rag_manager.ask_question(
            "What is the main purpose of this codebase?",
            test_path
        )
        print(f"✅ Answer: {answer}")
    except Exception as e:
        print(f"❌ Error asking question: {e}")
    
    # Test 4: Get code context
    print("\n4️⃣ Testing code context retrieval...")
    try:
        context = rag_manager.get_code_context("main function", test_path)
        print(f"✅ Retrieved {len(context)} relevant snippets")
    except Exception as e:
        print(f"❌ Error getting context: {e}")
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    test_rag_system() 