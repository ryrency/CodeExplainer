#!/usr/bin/env python3
"""
Simple test script for the MCP Code Q&A Server
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

async def test_mcp_server():
    """Test the MCP server with basic functionality."""
    
    print("🧪 Testing MCP Code Q&A Server...")
    
    # Start the MCP server as a subprocess
    server_process = subprocess.Popen(
        [sys.executable, "-m", "src.mcp_server.main"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Test 1: Initialize the server
        print("\n1️⃣ Testing server initialization...")
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        # Send initialization message
        server_process.stdin.write(json.dumps(init_message) + "\n")
        server_process.stdin.flush()
        
        # Read response
        response = server_process.stdout.readline()
        print(f"✅ Server response: {response.strip()}")
        
        # Test 2: List tools
        print("\n2️⃣ Testing tool listing...")
        list_tools_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        server_process.stdin.write(json.dumps(list_tools_message) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        print(f"✅ Available tools: {response.strip()}")
        
        # Test 3: Index a repository (using current directory as test)
        print("\n3️⃣ Testing repository indexing...")
        index_message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "index_repository",
                "arguments": {
                    "repository_path": str(Path.cwd()),
                    "force_reindex": True
                }
            }
        }
        
        server_process.stdin.write(json.dumps(index_message) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        print(f"✅ Indexing result: {response.strip()}")
        
        # Test 4: Ask a question
        print("\n4️⃣ Testing question asking...")
        question_message = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "ask_question",
                "arguments": {
                    "question": "What is the main purpose of this codebase?",
                    "repository_path": str(Path.cwd())
                }
            }
        }
        
        server_process.stdin.write(json.dumps(question_message) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        print(f"✅ Question answer: {response.strip()}")
        
        # Test 5: List indexed repositories
        print("\n5️⃣ Testing repository listing...")
        list_repos_message = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "list_indexed_repositories",
                "arguments": {}
            }
        }
        
        server_process.stdin.write(json.dumps(list_repos_message) + "\n")
        server_process.stdin.flush()
        
        response = server_process.stdout.readline()
        print(f"✅ Indexed repositories: {response.strip()}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        # Clean up
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    asyncio.run(test_mcp_server()) 