# MCP Code Q&A Server Architecture

## Overview

The MCP Code Q&A Server is a comprehensive system that provides intelligent code analysis and question-answering capabilities using RAG (Retrieval-Augmented Generation) technology. The system is designed to handle large repositories efficiently and provide accurate, context-aware answers to code-related questions.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Code Q&A Server                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   MCP Server    │  │  RAG System     │  │  Evaluation     │  │
│  │                 │  │                 │  │  Framework      │  │
│  │ • Tool Registry │  │ • Code Parser   │  │ • Metrics       │  │
│  │ • Request       │  │ • Vector Store  │  │ • Reports       │  │
│  │   Handler       │  │ • Retrieval     │  │ • Analysis      │  │
│  │ • Response      │  │ • Generation    │  │                 │  │
│  │   Generator     │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  LLM Agent      │  │  Configuration  │  │  Utilities      │  │
│  │                 │  │                 │  │                 │  │
│  │ • Repository    │  │ • Environment   │  │ • Logging       │  │
│  │   Analysis      │  │   Variables     │  │ • Error         │  │
│  │ • Architecture  │  │ • Model Config  │  │   Handling      │  │
│  │   Reports       │  │ • System        │  │ • File          │  │
│  │ • Pattern       │  │   Settings      │  │   Operations    │  │
│  │   Detection     │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. MCP Server (`src/mcp_server/`)

The MCP Server implements the Model Context Protocol and provides the main interface for code Q&A functionality.

**Key Features:**
- Tool registry and management
- Request/response handling
- Error handling and validation
- Async operation support

**Available Tools:**
- `ask_question`: Ask questions about indexed code
- `index_repository`: Index a repository for Q&A
- `get_code_context`: Retrieve relevant code snippets
- `list_indexed_repositories`: List all indexed repositories
- `get_repository_stats`: Get statistics about a repository

### 2. RAG System (`src/rag_system/`)

The RAG system is the core intelligence engine that handles code understanding and question answering.

**Components:**

#### RAG Manager (`rag_manager.py`)
- Coordinates all RAG operations
- Manages vector store interactions
- Handles repository indexing
- Orchestrates retrieval and generation

#### Code Parser (`src/code_parser/`)
- Intelligent code chunking into logical blocks
- Support for multiple programming languages
- AST-based parsing for Python
- Regex-based parsing for other languages
- Metadata extraction (function names, classes, etc.)

**Supported Languages:**
- Python (AST-based parsing)
- JavaScript/TypeScript
- Java
- C/C++
- Generic file parsing

### 3. Evaluation Framework (`src/evaluation/`)

Comprehensive evaluation system for measuring RAG system quality.

**Metrics:**
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU score
- Semantic similarity (cosine similarity of embeddings)
- Retrieval accuracy
- Answer length ratio
- Overall weighted score

**Features:**
- Automated evaluation against reference datasets
- Detailed performance reports
- Support for custom evaluation datasets
- Statistical analysis and insights

### 4. LLM Agent (`src/agent/`)

Tool-calling LLM agent for comprehensive repository analysis.

**Capabilities:**
- Repository architecture analysis
- Design pattern identification
- Dependency analysis
- Code quality assessment
- Improvement recommendations

**Tools:**
- Code question answering
- File structure analysis
- Dependency extraction
- Pattern identification

## Data Flow

### 1. Repository Indexing

```
Repository → Code Parser → Code Chunks → Vector Store → Metadata Storage
     ↓              ↓            ↓            ↓              ↓
  File System   AST/Regex   Logical Units   Embeddings   JSON Metadata
```

### 2. Question Answering

```
Question → Embedding → Vector Search → Context Retrieval → LLM Generation → Answer
    ↓          ↓            ↓              ↓                ↓            ↓
  Text     Vector      Similarity      Code Snippets    Prompt      Response
  Input    Query       Matching        + Metadata       Creation    Generation
```

### 3. Evaluation

```
Reference Q&A → RAG System → Generated Answer → Metrics Calculation → Report
      ↓              ↓              ↓                ↓                ↓
   Test Set      Question      LLM Response     ROUGE/BLEU/     Performance
   Loading      Processing     Generation       Similarity      Analysis
```

## Technology Stack

### Core Dependencies
- **MCP**: Model Context Protocol implementation
- **LangChain**: LLM orchestration and RAG framework
- **ChromaDB**: Vector database for similarity search
- **OpenAI**: Embeddings and LLM services
- **Tree-sitter**: Robust code parsing

### Evaluation Dependencies
- **ROUGE**: Text similarity metrics
- **BLEU**: Machine translation quality metrics
- **NLTK**: Natural language processing
- **scikit-learn**: Machine learning utilities
- **NumPy**: Numerical computing

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting

## Configuration

The system is configured through environment variables and the `Config` class:

```python
@dataclass
class Config:
    # API Keys
    openai_api_key: str
    anthropic_api_key: Optional[str]
    
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
```

## Performance Characteristics

### Scalability
- **Vector Store**: ChromaDB supports millions of embeddings
- **Chunking**: Intelligent parsing reduces storage requirements
- **Caching**: Repository metadata caching for fast lookups
- **Async Operations**: Non-blocking I/O for better performance

### Accuracy
- **Semantic Search**: Embedding-based similarity for context relevance
- **Logical Chunking**: Preserves code structure and relationships
- **Multi-metric Evaluation**: Comprehensive quality assessment
- **Context-Aware Generation**: LLM prompts include relevant code context

### Reliability
- **Error Handling**: Comprehensive exception handling
- **Validation**: Input validation and sanitization
- **Logging**: Detailed logging for debugging and monitoring
- **Testing**: Unit tests and integration tests

## Security Considerations

1. **API Key Management**: Secure storage of API keys
2. **Input Validation**: Sanitization of user inputs
3. **File System Access**: Controlled access to repository files
4. **Error Information**: Limited exposure of internal errors

## Deployment

### Requirements
- Python 3.8+
- OpenAI API key
- Sufficient disk space for vector store
- Memory for embedding operations

### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment variables
3. Run setup script: `./scripts/setup.sh`
4. Start MCP server: `python -m src.mcp_server.main`

## Future Enhancements

### Planned Features
- Support for more programming languages
- Advanced code analysis (complexity metrics, security scanning)
- Real-time collaboration features
- Integration with IDEs and code editors
- Custom model fine-tuning capabilities

### Performance Improvements
- Distributed vector store deployment
- Advanced caching strategies
- Optimized embedding models
- Parallel processing for large repositories

## Monitoring and Maintenance

### Health Checks
- Vector store connectivity
- API service availability
- Disk space monitoring
- Performance metrics tracking

### Maintenance Tasks
- Regular vector store optimization
- Metadata cleanup and validation
- Log rotation and archiving
- Dependency updates and security patches 