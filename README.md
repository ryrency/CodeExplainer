# MCP Code Q&A Server

A Model Context Protocol (MCP) server that provides intelligent code Q&A capabilities for local repositories using RAG (Retrieval-Augmented Generation) technology.

## Features

- **MCP Server**: Implements the Model Context Protocol for code Q&A
- **RAG System**: Advanced retrieval system with semantic search and code chunking
- **Code Analysis**: Intelligent parsing of Python code into logical blocks
- **Evaluation Framework**: Automated quality assessment using reference datasets
- **Repo Analysis Agent**: LLM agent for comprehensive repository analysis

## Project Structure

```
mcp-code-qa/
├── src/
│   ├── mcp_server/          # MCP server implementation
│   ├── rag_system/          # RAG components (chunking, indexing, retrieval)
│   ├── code_parser/         # Code parsing and analysis
│   ├── evaluation/          # Evaluation framework
│   └── agent/              # LLM agent for repo analysis
├── data/                   # Data storage and test datasets
├── tests/                  # Test files
├── docs/                   # Documentation
└── scripts/               # Utility scripts
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 3. Run the MCP Server

```bash
python -m src.mcp_server.main
```

### 4. Run Evaluation

```bash
python -m src.evaluation.evaluate
```

### 5. Run Repo Analysis Agent

```bash
python -m src.agent.repo_analyzer
```

## Usage

### MCP Server

The MCP server provides the following tools:
- `ask_question`: Ask questions about code in natural language
- `index_repository`: Index a new repository for Q&A
- `get_code_context`: Retrieve relevant code snippets

### Evaluation

The evaluation framework:
- Tests the RAG system against reference Q&A pairs
- Provides quantitative metrics (ROUGE, BLEU, semantic similarity)
- Generates detailed evaluation reports

### Repo Analysis Agent

The LLM agent:
- Analyzes repository architecture
- Identifies design patterns
- Lists external dependencies
- Generates comprehensive reports

## Architecture

### RAG System Components

1. **Code Chunking**: Intelligent parsing of code into logical blocks (functions, classes, methods)
2. **Vector Storage**: ChromaDB for efficient similarity search
3. **Retrieval**: Semantic search with context-aware ranking
4. **Generation**: LLM-powered answer generation with retrieved context

### Code Parser

- Uses tree-sitter for robust Python parsing
- Extracts function definitions, class definitions, and method implementations
- Preserves code structure and relationships

### Evaluation Metrics

- ROUGE scores for text similarity
- BLEU scores for answer quality
- Semantic similarity using embeddings
- Custom code-specific metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License 