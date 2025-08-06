# MCP Code Q&A Server

A Model Context Protocol (MCP) server that provides intelligent code Q&A capabilities for local repositories using RAG (Retrieval-Augmented Generation) technology.

## 🚀 Features

- **MCP Server**: Implements the Model Context Protocol for code Q&A
- **REST API**: FastAPI-based REST API for easy integration
- **RAG System**: Advanced retrieval system with semantic search and code chunking
- **Code Analysis**: Intelligent parsing of Python, JavaScript, TypeScript, Java, and C++ code
- **Evaluation Framework**: Automated quality assessment using reference datasets
- **Repo Analysis Agent**: LLM agent for comprehensive repository analysis

## 📁 Project Structure

```
mcp-code-qa/
├── src/
│   ├── mcp_server/          # MCP server implementation
│   ├── rag_system/          # RAG components (chunking, indexing, retrieval)
│   ├── code_parser/         # Code parsing and analysis
│   ├── evaluation/          # Evaluation framework
│   ├── agent/              # LLM agent for repo analysis
│   └── utils/              # Configuration and utilities
├── data/                   # Data storage and test datasets
├── tests/                  # Test files
├── docs/                   # Documentation
├── scripts/               # Utility scripts
├── api_wrapper.py         # FastAPI REST server
└── requirements.txt       # Python dependencies
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys))

### 1. Clone and Navigate

```bash
git clone <your-repo-url>
cd mcp-code-qa
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model Configuration (optional - defaults shown)
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4-turbo-preview

# RAG Configuration (optional - defaults shown)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_RESULTS=5

# Vector Store Configuration (optional - defaults shown)
VECTOR_STORE_PATH=data/vector_store
```

**⚠️ Security Note**: Never commit your `.env` file to version control. The `.gitignore` file is already configured to exclude it.

### 5. Run the API Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start the API server
python api_wrapper.py
```

The server will start on `http://localhost:8000` with automatic reload enabled.

## 📖 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### API Information
```bash
curl http://localhost:8000/
```

### Index a Repository
```bash
curl -X POST http://localhost:8000/index-repository \
  -H "Content-Type: application/json" \
  -d '{"repository_path": "/path/to/your/repository", "force_reindex": false}'
```

### Ask Questions About Code
```bash
curl -X POST http://localhost:8000/ask-question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this project about?", "repository_path": "/path/to/your/repository"}'
```

### Get Code Context
```bash
curl -X POST http://localhost:8000/get-context \
  -H "Content-Type: application/json" \
  -d '{"query": "main function", "repository_path": "/path/to/your/repository", "max_results": 5}'
```

### List Indexed Repositories
```bash
curl -X GET http://localhost:8000/list-repositories
```

### Get Repository Statistics
```bash
curl -X GET http://localhost:8000/repository-stats/your-repository-path
```

## 🌐 Web Interface

Access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Alternative: Run MCP Server Directly

If you want to run the MCP server directly (for use with MCP clients):

```bash
# Run the MCP server
python -m src.mcp_server.main
```

## 📊 Example Usage

### 1. Start the server:
```bash
source venv/bin/activate
python api_wrapper.py
```

### 2. Index your repository:
```bash
curl -X POST http://localhost:8000/index-repository \
  -H "Content-Type: application/json" \
  -d '{"repository_path": "/path/to/your/project", "force_reindex": true}'
```

### 3. Ask questions:
```bash
curl -X POST http://localhost:8000/ask-question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this project about?", "repository_path": "/path/to/your/project"}'
```

## 🔒 Security Considerations

### API Key Security
- **Never share your API key** - it's like a password
- **Never commit `.env` files** to version control
- **Monitor your API usage** to avoid unexpected charges
- **Rotate keys regularly** for better security

### Safe Sharing
When sharing this project with others:
1. **Remove your API key** from any shared files
2. **Provide setup instructions** for getting their own API key
3. **Use `.env.example`** as a template (without real keys)

## 🐛 Troubleshooting

### Common Issues

**Port already in use:**
```bash
pkill -f "python api_wrapper.py"
```

**API key issues:**
- Ensure your `.env` file is properly configured
- Check that the API key is valid and has sufficient credits

**Import errors:**
```bash
pip install -r requirements.txt
```

**Large files in vector store:**
- The vector store may contain large files that exceed GitHub's limits
- Consider using `.gitignore` to exclude `data/vector_store/` from version control

## 🏗️ Architecture

### RAG System Components

1. **Code Chunking**: Intelligent parsing of code into logical blocks (functions, classes, methods)
2. **Vector Storage**: ChromaDB for efficient similarity search
3. **Retrieval**: Semantic search with context-aware ranking
4. **Generation**: LLM-powered answer generation with retrieved context

### Supported Languages

- **Python**: Full AST-based parsing
- **JavaScript/TypeScript**: Regex-based parsing
- **Java**: Regex-based parsing
- **C++**: Regex-based parsing

### Code Parser

- Uses AST for Python files
- Regex patterns for other languages
- Extracts function definitions, class definitions, and method implementations
- Preserves code structure and relationships

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/
```

### Run Evaluation
```bash
python -m src.evaluation.evaluate
```

### Run Repo Analysis Agent
```bash
python -m src.agent.repo_analyzer
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License 