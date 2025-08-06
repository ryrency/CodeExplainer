#!/usr/bin/env python3
"""
Debug script to test the code parser
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_parser.parser import CodeParser

def test_parser():
    """Test the code parser with a simple file."""
    
    parser = CodeParser()
    
    # Test with a simple Python file
    test_file = Path("src/utils/config.py")
    
    print(f"Testing parser with: {test_file}")
    print(f"File exists: {test_file.exists()}")
    
    if test_file.exists():
        chunks = parser.parse_file(test_file)
        print(f"Parsed {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}:")
            print(f"  Type: {type(chunk)}")
            print(f"  Chunk type: {chunk.chunk_type if hasattr(chunk, 'chunk_type') else 'N/A'}")
            print(f"  File: {chunk.file_path if hasattr(chunk, 'file_path') else 'N/A'}")
            print(f"  Has metadata: {hasattr(chunk, 'metadata')}")
            if hasattr(chunk, 'metadata'):
                print(f"  Metadata: {chunk.metadata}")
            print()

def test_directory():
    """Test directory parsing."""
    
    parser = CodeParser()
    
    # Test with src directory
    test_dir = Path("src")
    
    print(f"Testing directory parsing with: {test_dir}")
    print(f"Directory exists: {test_dir.exists()}")
    
    if test_dir.exists():
        chunks = parser.parse_directory(test_dir)
        print(f"Parsed {len(chunks)} chunks from directory")
        
        # Check for any non-CodeChunk objects
        invalid_chunks = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, type(parser.parse_file(Path("src/utils/config.py"))[0])):
                invalid_chunks.append((i, chunk))
        
        if invalid_chunks:
            print(f"Found {len(invalid_chunks)} invalid chunks:")
            for i, chunk in invalid_chunks:
                print(f"  Chunk {i}: {type(chunk)} - {str(chunk)[:100]}")
        else:
            print("All chunks are valid CodeChunk objects")

if __name__ == "__main__":
    print("=== Testing single file parsing ===")
    test_parser()
    print("\n=== Testing directory parsing ===")
    test_directory() 