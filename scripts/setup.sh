#!/bin/bash

# Setup script for MCP Code Q&A Server

echo "🚀 Setting up MCP Code Q&A Server..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | sed 's/Python //' | cut -d. -f1,2)
major_version=$(echo $python_version | cut -d. -f1)
minor_version=$(echo $python_version | cut -d. -f2)

# Check if version is >= 3.8
if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 8 ]]; then
    echo "❌ Python 3.8 or higher is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/vector_store
mkdir -p data/evaluation
mkdir -p data/analysis
mkdir -p logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔧 Creating .env file..."
    cat > .env << EOF
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
EOF
    echo "⚠️  Please update .env file with your API keys!"
fi

# Download NLTK data for evaluation
echo "📥 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" 2>/dev/null || echo "NLTK data download skipped (will be downloaded when needed)"

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your API keys"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run the MCP server: python -m src.mcp_server.main"
echo "4. Test with evaluation: python -m src.evaluation.evaluate --repository /path/to/repo --sample"
echo ""
echo "For more information, see README.md" 