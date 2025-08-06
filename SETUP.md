# 🚀 Setup Guide for MCP Code Q&A Server

This guide will help you set up and run the MCP Code Q&A Server on your local machine.

## 📋 Prerequisites

- **Python 3.8 or higher**
- **OpenAI API key** (get one at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys))
- **Git** (for cloning the repository)

## 🔧 Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd mcp-code-qa
```

### 2. Create Virtual Environment

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

Create a `.env` file in the project root with the following content:

```bash
# API Keys (replace with your actual keys)
# Get your OpenAI API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Anthropic API key (alternative to OpenAI)
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

**⚠️ Important**: Replace `your_openai_api_key_here` with your actual OpenAI API key.

### 5. Run the Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start the API server
python api_wrapper.py
```

The server will start on `http://localhost:8000` with automatic reload enabled.

## 🔒 Security Best Practices

### API Key Security
- **Never share your API key** - treat it like a password
- **Never commit `.env` files** to version control
- **Monitor your API usage** to avoid unexpected charges
- **Rotate keys regularly** for better security

### Safe Sharing
When sharing this project with others:
1. **Remove your API key** from any shared files
2. **Provide setup instructions** for getting their own API key
3. **Use this template** for the `.env` file (without real keys)

## 🧪 Testing Your Setup

### 1. Check Server Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "openai_key_configured": true,
  "embedding_model": "text-embedding-3-small",
  "llm_model": "gpt-4-turbo-preview"
}
```

### 2. Index a Repository
```bash
curl -X POST http://localhost:8000/index-repository \
  -H "Content-Type: application/json" \
  -d '{"repository_path": "/path/to/your/project", "force_reindex": true}'
```

### 3. Ask a Question
```bash
curl -X POST http://localhost:8000/ask-question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this project about?", "repository_path": "/path/to/your/project"}'
```

## 🌐 Web Interface

Access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐛 Troubleshooting

### Common Issues

**Port already in use:**
```bash
pkill -f "python api_wrapper.py"
```

**API key not configured:**
- Check that your `.env` file exists and contains the correct API key
- Ensure the virtual environment is activated

**Import errors:**
```bash
pip install -r requirements.txt
```

**Large files in vector store:**
- The vector store may contain large files that exceed GitHub's limits
- Consider using `.gitignore` to exclude `data/vector_store/` from version control

### Getting Help

If you encounter issues:
1. Check the server logs for error messages
2. Verify your API key is valid and has sufficient credits
3. Ensure all dependencies are installed correctly
4. Check that the virtual environment is activated

## 📚 Next Steps

Once your server is running:
1. **Index your repositories** using the API
2. **Ask questions** about your code
3. **Explore the web interface** at http://localhost:8000/docs
4. **Check out the examples** in the main README

## 🔗 Useful Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/) 