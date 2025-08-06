# Setup and Usage Guide

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (for cloning the repository)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd mcp-code-qa

# Run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Activate virtual environment
source venv/bin/activate
```

### 3. Configuration

Edit the `.env` file with your API keys:

```bash
# API Keys (replace with your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4-turbo-preview

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_RESULTS=5

# Vector Store Configuration
VECTOR_STORE_PATH=data/vector_store
```

### 4. Start the MCP Server

```bash
# Start the MCP server
python -m src.mcp_server.main
```

## Usage Examples

### 1. Indexing a Repository

```bash
# Using the MCP server tools
python -c "
import asyncio
from src.rag_system.rag_manager import RAGManager
from src.utils.config import Config

async def index_repo():
    config = Config()
    rag_manager = RAGManager(config)
    stats = await rag_manager.index_repository('/path/to/your/repo')
    print(f'Indexed {stats[\"chunks_created\"]} chunks')

asyncio.run(index_repo())
"
```

### 2. Asking Questions

```bash
# Using the MCP server tools
python -c "
import asyncio
from src.rag_system.rag_manager import RAGManager
from src.utils.config import Config

async def ask_question():
    config = Config()
    rag_manager = RAGManager(config)
    answer = await rag_manager.ask_question(
        'What does the main function do?',
        '/path/to/your/repo'
    )
    print(answer)

asyncio.run(ask_question())
"
```

### 3. Running Evaluation

```bash
# Evaluate with sample dataset
python -m src.evaluation.evaluate \
    --repository /path/to/your/repo \
    --sample \
    --output data/evaluation/results.json \
    --report data/evaluation/report.txt

# Evaluate with custom dataset
python -m src.evaluation.evaluate \
    --repository /path/to/your/repo \
    --dataset data/evaluation/custom_dataset.json \
    --output data/evaluation/results.json \
    --report data/evaluation/report.txt
```

### 4. Repository Analysis

```bash
# Run comprehensive repository analysis
python -m src.agent.repo_analyzer \
    --repository /path/to/your/repo \
    --output data/analysis/report.txt \
    --json data/analysis/results.json
```

## MCP Server Tools

The MCP server provides the following tools:

### 1. `ask_question`

Ask questions about code in the indexed repository.

**Parameters:**
- `question` (string): The question to ask about the code
- `repository_path` (string): Path to the repository to query

**Example:**
```json
{
  "name": "ask_question",
  "arguments": {
    "question": "What does the main function do?",
    "repository_path": "/path/to/repo"
  }
}
```

### 2. `index_repository`

Index a repository for Q&A by parsing and storing code chunks.

**Parameters:**
- `repository_path` (string): Path to the repository to index
- `force_reindex` (boolean, optional): Force reindexing even if already indexed

**Example:**
```json
{
  "name": "index_repository",
  "arguments": {
    "repository_path": "/path/to/repo",
    "force_reindex": false
  }
}
```

### 3. `get_code_context`

Retrieve relevant code snippets for a given query.

**Parameters:**
- `query` (string): Query to find relevant code snippets
- `repository_path` (string): Path to the repository to search
- `max_results` (integer, optional): Maximum number of results to return

**Example:**
```json
{
  "name": "get_code_context",
  "arguments": {
    "query": "database connection",
    "repository_path": "/path/to/repo",
    "max_results": 5
  }
}
```

### 4. `list_indexed_repositories`

List all currently indexed repositories.

**Example:**
```json
{
  "name": "list_indexed_repositories",
  "arguments": {}
}
```

### 5. `get_repository_stats`

Get statistics about an indexed repository.

**Parameters:**
- `repository_path` (string): Path to the repository

**Example:**
```json
{
  "name": "get_repository_stats",
  "arguments": {
    "repository_path": "/path/to/repo"
  }
}
```

## Evaluation Framework

### Metrics Explained

1. **ROUGE Scores**: Measure text similarity between generated and reference answers
   - ROUGE-1: Unigram overlap
   - ROUGE-2: Bigram overlap
   - ROUGE-L: Longest common subsequence

2. **BLEU Score**: Measures the quality of machine translation (adapted for Q&A)

3. **Semantic Similarity**: Cosine similarity between embeddings of generated and reference answers

4. **Retrieval Accuracy**: Measures how well the system retrieves relevant code snippets

5. **Overall Score**: Weighted combination of all metrics

### Creating Custom Evaluation Datasets

Create a JSON file with the following format:

```json
[
  {
    "question": "Your question here?",
    "answer": "Reference answer here."
  },
  {
    "question": "Another question?",
    "answer": "Another reference answer."
  }
]
```

### Running Evaluations

```bash
# Basic evaluation with sample data
python -m src.evaluation.evaluate --repository /path/to/repo --sample

# Full evaluation with custom dataset
python -m src.evaluation.evaluate \
    --repository /path/to/repo \
    --dataset path/to/dataset.json \
    --output results.json \
    --report report.txt

# Force reindexing before evaluation
python -m src.evaluation.evaluate \
    --repository /path/to/repo \
    --sample \
    --force-reindex
```

## Repository Analysis Agent

### Features

The LLM agent provides comprehensive repository analysis including:

- **Architecture Summary**: Overall structure and design
- **Design Patterns**: Identification of common patterns
- **External Dependencies**: Analysis of libraries and frameworks
- **Main Components**: Key components and their responsibilities
- **Code Quality Insights**: Assessment of code quality
- **Recommendations**: Suggestions for improvement

### Usage

```bash
# Basic analysis
python -m src.agent.repo_analyzer --repository /path/to/repo

# Custom output paths
python -m src.agent.repo_analyzer \
    --repository /path/to/repo \
    --output analysis_report.txt \
    --json analysis_results.json
```

## Advanced Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `LLM_MODEL` | LLM model name | `gpt-4-turbo-preview` |
| `CHUNK_SIZE` | Code chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` |
| `MAX_RETRIEVAL_RESULTS` | Max retrieval results | `5` |
| `VECTOR_STORE_PATH` | Vector store path | `data/vector_store` |

### Custom Configuration

You can modify the configuration by editing `src/utils/config.py` or setting environment variables.

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Error: OpenAI API key is required
   ```
   Solution: Set the `OPENAI_API_KEY` environment variable

2. **Repository Not Found**
   ```
   Error: Repository path does not exist
   ```
   Solution: Check the repository path and ensure it exists

3. **Vector Store Errors**
   ```
   Error: Vector store initialization failed
   ```
   Solution: Check disk space and permissions for the vector store directory

4. **Memory Issues**
   ```
   Error: Out of memory during indexing
   ```
   Solution: Reduce chunk size or process smaller repositories

### Debug Mode

Enable debug logging by setting the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

1. **Reduce Chunk Size**: For memory-constrained environments
2. **Increase Chunk Overlap**: For better context preservation
3. **Use Smaller Models**: For faster processing
4. **Batch Processing**: For large repositories

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_code_parser.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

### Test Coverage

The test suite covers:
- Code parsing functionality
- RAG system operations
- Evaluation metrics
- Configuration management
- Error handling

## Deployment

### Production Setup

1. **Environment**: Use a production Python environment
2. **API Keys**: Store securely using environment variables or secrets management
3. **Monitoring**: Set up logging and monitoring
4. **Backup**: Regular backups of vector store and metadata
5. **Security**: Implement proper access controls

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "src.mcp_server.main"]
```

### Scaling Considerations

1. **Vector Store**: Consider distributed ChromaDB for large-scale deployments
2. **Caching**: Implement Redis for caching frequently accessed data
3. **Load Balancing**: Use multiple MCP server instances
4. **Storage**: Use cloud storage for vector store data

## Support and Contributing

### Getting Help

1. Check the documentation in the `docs/` directory
2. Review the test files for usage examples
3. Check the issue tracker for known problems
4. Create a new issue for bugs or feature requests

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

The project uses:
- **Black**: Code formatting
- **Flake8**: Code linting
- **Pytest**: Testing framework

Run the formatters before submitting:

```bash
black src/ tests/
flake8 src/ tests/
``` 