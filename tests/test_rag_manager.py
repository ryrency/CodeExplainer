"""
Unit tests for the RAG manager.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.rag_system.rag_manager import RAGManager
from src.utils.config import Config
from src.code_parser.parser import CodeChunk

class TestRAGManager:
    """Test cases for RAGManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock config
        self.config = Mock(spec=Config)
        self.config.openai_api_key = "test_key"
        self.config.embedding_model = "text-embedding-3-small"
        self.config.llm_model = "gpt-4-turbo-preview"
        self.config.vector_store_path = tempfile.mkdtemp()
        self.config.collection_name = "test_collection"
        self.config.max_retrieval_results = 5
        
        # Mock the vector store and embeddings
        with patch('src.rag_system.rag_manager.Chroma'), \
             patch('src.rag_system.rag_manager.OpenAIEmbeddings'), \
             patch('src.rag_system.rag_manager.ChatOpenAI'), \
             patch('src.rag_system.rag_manager.chromadb.PersistentClient'):
            
            self.rag_manager = RAGManager(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if hasattr(self.config, 'vector_store_path'):
            shutil.rmtree(self.config.vector_store_path, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_index_repository(self):
        """Test indexing a repository."""
        # Create a temporary repository
        temp_repo = tempfile.mkdtemp()
        repo_path = Path(temp_repo)
        
        # Create some test files
        (repo_path / "main.py").write_text("def main():\n    print('Hello')\n")
        (repo_path / "utils.py").write_text("import os\ndef helper():\n    pass\n")
        
        # Mock the code parser
        mock_chunks = [
            CodeChunk(
                content="def main():\n    print('Hello')\n",
                chunk_type="function",
                file_path=str(repo_path / "main.py"),
                start_line=1,
                end_line=2,
                metadata={"function_name": "main"}
            ),
            CodeChunk(
                content="import os\ndef helper():\n    pass\n",
                chunk_type="function",
                file_path=str(repo_path / "utils.py"),
                start_line=2,
                end_line=3,
                metadata={"function_name": "helper"}
            )
        ]
        
        with patch.object(self.rag_manager.code_parser, 'parse_directory', return_value=mock_chunks):
            stats = await self.rag_manager.index_repository(temp_repo)
        
        assert stats['files_processed'] == 2
        assert stats['chunks_created'] == 2
        assert 'indexing_time' in stats
        assert self.rag_manager.is_repository_indexed(temp_repo)
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_repo)
    
    @pytest.mark.asyncio
    async def test_ask_question(self):
        """Test asking a question."""
        # Mock repository metadata
        repo_path = "/test/repo"
        self.rag_manager.repo_metadata[repo_path] = {
            'repository_path': repo_path,
            'files_processed': 1,
            'chunks_created': 1
        }
        
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = "This is a test answer"
        self.rag_manager.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Mock get_relevant_snippets
        mock_snippets = [
            {
                'code': 'def test():\n    pass',
                'file_path': 'test.py',
                'chunk_type': 'function',
                'start_line': 1,
                'end_line': 2,
                'score': 0.8,
                'metadata': {}
            }
        ]
        self.rag_manager.get_relevant_snippets = AsyncMock(return_value=mock_snippets)
        
        answer = await self.rag_manager.ask_question("What does the main function do?", repo_path)
        
        assert answer == "This is a test answer"
        self.rag_manager.get_relevant_snippets.assert_called_once()
        self.rag_manager.llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ask_question_unindexed_repo(self):
        """Test asking a question for an unindexed repository."""
        with pytest.raises(ValueError, match="Repository.*is not indexed"):
            await self.rag_manager.ask_question("Test question", "/unindexed/repo")
    
    @pytest.mark.asyncio
    async def test_get_relevant_snippets(self):
        """Test getting relevant code snippets."""
        repo_path = "/test/repo"
        
        # Mock vector store search results
        mock_doc = Mock()
        mock_doc.page_content = "def test():\n    pass"
        mock_doc.metadata = {
            'file_path': 'test.py',
            'chunk_type': 'function',
            'start_line': 1,
            'end_line': 2
        }
        
        self.rag_manager.vector_store.similarity_search_with_score = Mock(
            return_value=[(mock_doc, 0.8)]
        )
        
        snippets = await self.rag_manager.get_relevant_snippets("test function", repo_path, 3)
        
        assert len(snippets) == 1
        assert snippets[0]['code'] == "def test():\n    pass"
        assert snippets[0]['file_path'] == 'test.py'
        assert snippets[0]['score'] == 0.8
    
    def test_is_repository_indexed(self):
        """Test checking if a repository is indexed."""
        repo_path = "/test/repo"
        
        # Test unindexed repository
        assert not self.rag_manager.is_repository_indexed(repo_path)
        
        # Test indexed repository
        self.rag_manager.repo_metadata[repo_path] = {}
        assert self.rag_manager.is_repository_indexed(repo_path)
    
    def test_list_indexed_repositories(self):
        """Test listing indexed repositories."""
        # Add some mock repositories
        self.rag_manager.repo_metadata = {
            "/repo1": {"chunks_created": 10, "last_indexed": "2023-01-01", "files_processed": 5},
            "/repo2": {"chunks_created": 20, "last_indexed": "2023-01-02", "files_processed": 8}
        }
        
        # Mock vector store get method
        self.rag_manager.vector_store.get = Mock(return_value={'ids': ['id1', 'id2']})
        
        repositories = self.rag_manager.list_indexed_repositories()
        
        assert len(repositories) == 2
        assert any(r['path'] == "/repo1" for r in repositories)
        assert any(r['path'] == "/repo2" for r in repositories)
    
    def test_get_repository_stats(self):
        """Test getting repository statistics."""
        repo_path = "/test/repo"
        
        # Test unindexed repository
        assert self.rag_manager.get_repository_stats(repo_path) is None
        
        # Test indexed repository
        self.rag_manager.repo_metadata[repo_path] = {
            'files_processed': 5,
            'chunks_created': 10,
            'file_types': ['.py', '.js'],
            'chunk_types': ['function', 'class'],
            'last_indexed': '2023-01-01',
            'indexing_time': 5.5
        }
        
        # Mock vector store get method
        self.rag_manager.vector_store.get = Mock(return_value={'ids': ['id1', 'id2']})
        
        stats = self.rag_manager.get_repository_stats(repo_path)
        
        assert stats is not None
        assert stats['total_files'] == 5
        assert stats['total_chunks'] == 2
        assert '.py' in stats['file_types']
        assert 'function' in stats['chunk_types']
    
    @pytest.mark.asyncio
    async def test_delete_repository(self):
        """Test deleting a repository from the index."""
        repo_path = "/test/repo"
        
        # Add repository to metadata
        self.rag_manager.repo_metadata[repo_path] = {}
        
        # Mock vector store operations
        self.rag_manager.vector_store.get = Mock(return_value={'ids': ['id1']})
        self.rag_manager.vector_store.delete = Mock()
        
        result = await self.rag_manager.delete_repository(repo_path)
        
        assert result is True
        assert repo_path not in self.rag_manager.repo_metadata
        self.rag_manager.vector_store.delete.assert_called_once_with(ids=['id1'])
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_repository(self):
        """Test deleting a repository that doesn't exist."""
        result = await self.rag_manager.delete_repository("/nonexistent/repo")
        assert result is False
    
    def test_create_context_from_snippets(self):
        """Test creating context from code snippets."""
        snippets = [
            {
                'code': 'def test():\n    pass',
                'file_path': 'test.py',
                'chunk_type': 'function',
                'start_line': 1,
                'end_line': 2,
                'score': 0.8,
                'metadata': {}
            }
        ]
        
        context = self.rag_manager._create_context_from_snippets(snippets)
        
        assert "Code Snippet 1:" in context
        assert "File: test.py" in context
        assert "Type: function" in context
        assert "def test():" in context
        assert "0.800" in context  # Score
    
    def test_create_qa_prompt(self):
        """Test creating Q&A prompt."""
        question = "What does this function do?"
        context = "def test():\n    pass"
        
        prompt = self.rag_manager._create_qa_prompt(question, context)
        
        assert question in prompt
        assert context in prompt
        assert "You are an expert code analyst" in prompt
        assert "Instructions:" in prompt

if __name__ == "__main__":
    pytest.main([__file__]) 