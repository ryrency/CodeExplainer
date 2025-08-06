"""
Main MCP Server implementation for Code Q&A
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
)

from ..rag_system.rag_manager import RAGManager
from ..code_parser.parser import CodeParser
from ..utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global components
config = Config()
rag_manager = RAGManager(config)
code_parser = CodeParser()

# Create MCP server
server = Server("code-qa-server")

@server.list_tools()
async def handle_list_tools() -> ListToolsResult:
    """List available tools."""
    tools = [
        Tool(
            name="ask_question",
            description="Ask a question about code in the indexed repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the code"
                    },
                    "repository_path": {
                        "type": "string",
                        "description": "Path to the repository to query"
                    }
                },
                "required": ["question", "repository_path"]
            }
        ),
        Tool(
            name="index_repository",
            description="Index a repository for Q&A by parsing and storing code chunks",
            inputSchema={
                "type": "object",
                "properties": {
                    "repository_path": {
                        "type": "string",
                        "description": "Path to the repository to index"
                    },
                    "force_reindex": {
                        "type": "boolean",
                        "description": "Force reindexing even if already indexed",
                        "default": False
                    }
                },
                "required": ["repository_path"]
            }
        ),
        Tool(
            name="get_code_context",
            description="Retrieve relevant code snippets for a given query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to find relevant code snippets"
                    },
                    "repository_path": {
                        "type": "string",
                        "description": "Path to the repository to search"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query", "repository_path"]
            }
        ),
        Tool(
            name="list_indexed_repositories",
            description="List all currently indexed repositories",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_repository_stats",
            description="Get statistics about an indexed repository",
            inputSchema={
                "type": "object",
                "properties": {
                    "repository_path": {
                        "type": "string",
                        "description": "Path to the repository"
                    }
                },
                "required": ["repository_path"]
            }
        )
    ]
    return ListToolsResult(tools=tools)

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
    """Handle tool calls."""
    try:
        if name == "ask_question":
            return await handle_ask_question(arguments)
        elif name == "index_repository":
            return await handle_index_repository(arguments)
        elif name == "get_code_context":
            return await handle_get_code_context(arguments)
        elif name == "list_indexed_repositories":
            return await handle_list_indexed_repositories(arguments)
        elif name == "get_repository_stats":
            return await handle_get_repository_stats(arguments)
        else:
            return CallToolResult(
                content=[{"type": "text", "text": f"Unknown tool: {name}"}],
                isError=True
            )
    except Exception as e:
        logger.error(f"Error in tool call {name}: {str(e)}")
        return CallToolResult(
            content=[{"type": "text", "text": f"Error: {str(e)}"}],
            isError=True
        )

async def handle_ask_question(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle ask_question tool call."""
    question = arguments.get("question")
    repository_path = arguments.get("repository_path")
    
    if not question or not repository_path:
        return CallToolResult(
            content=[{"type": "text", "text": "Missing required arguments: question and repository_path"}],
            isError=True
        )
    
    try:
        # Check if repository is indexed
        if not rag_manager.is_repository_indexed(repository_path):
            return CallToolResult(
                content=[{"type": "text", "text": f"Repository {repository_path} is not indexed. Please index it first using the index_repository tool."}],
                isError=True
            )
        
        # Get answer using RAG system
        answer = await rag_manager.ask_question(question, repository_path)
        
        return CallToolResult(
            content=[{"type": "text", "text": answer}]
        )
    except Exception as e:
        logger.error(f"Error asking question: {str(e)}")
        return CallToolResult(
            content=[{"type": "text", "text": f"Error processing question: {str(e)}"}],
            isError=True
        )

async def handle_index_repository(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle index_repository tool call."""
    repository_path = arguments.get("repository_path")
    force_reindex = arguments.get("force_reindex", False)
    
    if not repository_path:
        return CallToolResult(
            content=[{"type": "text", "text": "Missing required argument: repository_path"}],
            isError=True
        )
    
    try:
        # Check if repository exists
        if not Path(repository_path).exists():
            return CallToolResult(
                content=[{"type": "text", "text": f"Repository path {repository_path} does not exist"}],
                isError=True
            )
        
        # Index the repository
        stats = await rag_manager.index_repository(repository_path, force_reindex)
        
        result_text = f"Successfully indexed repository: {repository_path}\n"
        result_text += f"Files processed: {stats['files_processed']}\n"
        result_text += f"Code chunks created: {stats['chunks_created']}\n"
        result_text += f"Indexing time: {stats['indexing_time']:.2f} seconds"
        
        return CallToolResult(
            content=[{"type": "text", "text": result_text}]
        )
    except Exception as e:
        logger.error(f"Error indexing repository: {str(e)}")
        return CallToolResult(
            content=[{"type": "text", "text": f"Error indexing repository: {str(e)}"}],
            isError=True
        )

async def handle_get_code_context(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle get_code_context tool call."""
    query = arguments.get("query")
    repository_path = arguments.get("repository_path")
    max_results = arguments.get("max_results", 5)
    
    if not query or not repository_path:
        return CallToolResult(
            content=[{"type": "text", "text": "Missing required arguments: query and repository_path"}],
            isError=True
        )
    
    try:
        # Check if repository is indexed
        if not rag_manager.is_repository_indexed(repository_path):
            return CallToolResult(
                content=[{"type": "text", "text": f"Repository {repository_path} is not indexed. Please index it first using the index_repository tool."}],
                isError=True
            )
        
        # Get relevant code snippets
        snippets = await rag_manager.get_relevant_snippets(query, repository_path, max_results)
        
        if not snippets:
            return CallToolResult(
                content=[{"type": "text", "text": "No relevant code snippets found for the query."}]
            )
        
        result_text = "Relevant code snippets:\n\n"
        for i, snippet in enumerate(snippets, 1):
            result_text += f"--- Snippet {i} ---\n"
            result_text += f"File: {snippet['file_path']}\n"
            result_text += f"Score: {snippet['score']:.3f}\n"
            result_text += f"Code:\n{snippet['code']}\n\n"
        
        return CallToolResult(
            content=[{"type": "text", "text": result_text}]
        )
    except Exception as e:
        logger.error(f"Error getting code context: {str(e)}")
        return CallToolResult(
            content=[{"type": "text", "text": f"Error retrieving code context: {str(e)}"}],
            isError=True
        )

async def handle_list_indexed_repositories(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle list_indexed_repositories tool call."""
    try:
        repositories = rag_manager.list_indexed_repositories()
        
        if not repositories:
            return CallToolResult(
                content=[{"type": "text", "text": "No repositories are currently indexed."}]
            )
        
        result_text = "Indexed repositories:\n"
        for repo in repositories:
            result_text += f"- {repo['path']} (chunks: {repo['chunk_count']})\n"
        
        return CallToolResult(
            content=[{"type": "text", "text": result_text}]
        )
    except Exception as e:
        logger.error(f"Error listing repositories: {str(e)}")
        return CallToolResult(
            content=[{"type": "text", "text": f"Error listing repositories: {str(e)}"}],
            isError=True
        )

async def handle_get_repository_stats(arguments: Dict[str, Any]) -> CallToolResult:
    """Handle get_repository_stats tool call."""
    repository_path = arguments.get("repository_path")
    
    if not repository_path:
        return CallToolResult(
            content=[{"type": "text", "text": "Missing required argument: repository_path"}],
            isError=True
        )
    
    try:
        stats = rag_manager.get_repository_stats(repository_path)
        
        if not stats:
            return CallToolResult(
                content=[{"type": "text", "text": f"Repository {repository_path} is not indexed or not found."}]
            )
        
        result_text = f"Repository Statistics for {repository_path}:\n"
        result_text += f"Total files: {stats['total_files']}\n"
        result_text += f"Total code chunks: {stats['total_chunks']}\n"
        result_text += f"File types: {', '.join(stats['file_types'])}\n"
        result_text += f"Index size: {stats['index_size_mb']:.2f} MB\n"
        result_text += f"Last indexed: {stats['last_indexed']}"
        
        return CallToolResult(
            content=[{"type": "text", "text": result_text}]
        )
    except Exception as e:
        logger.error(f"Error getting repository stats: {str(e)}")
        return CallToolResult(
            content=[{"type": "text", "text": f"Error getting repository stats: {str(e)}"}],
            isError=True
        )

async def main():
    """Main entry point for the MCP server."""
    # Initialize the server
    async with stdio_server() as (read_stream, write_stream):
        # Create a simple notification options object
        from mcp.server import NotificationOptions
        notification_options = NotificationOptions(tools_changed=False)
        
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="code-qa-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=notification_options,
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 