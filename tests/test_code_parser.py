"""
Unit tests for the code parser.
"""

import pytest
from pathlib import Path
import tempfile
import os

from src.code_parser.parser import CodeParser, CodeChunk

class TestCodeParser:
    """Test cases for CodeParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CodeParser()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_parse_python_function(self):
        """Test parsing a Python function."""
        code = """
def hello_world(name: str) -> str:
    \"\"\"Say hello to the world.\"\"\"
    return f"Hello, {name}!"
"""
        
        temp_file = Path(self.temp_dir) / "test.py"
        with open(temp_file, 'w') as f:
            f.write(code)
        
        chunks = self.parser.parse_file(temp_file)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_type == 'function'
        assert chunk.metadata['function_name'] == 'hello_world'
        assert 'name' in chunk.metadata['arguments']
    
    def test_parse_python_class(self):
        """Test parsing a Python class."""
        code = """
class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x: int) -> int:
        self.value += x
        return self.value
"""
        
        temp_file = Path(self.temp_dir) / "calculator.py"
        with open(temp_file, 'w') as f:
            f.write(code)
        
        chunks = self.parser.parse_file(temp_file)
        
        assert len(chunks) >= 1
        class_chunks = [c for c in chunks if c.chunk_type == 'class']
        assert len(class_chunks) == 1
        
        class_chunk = class_chunks[0]
        assert class_chunk.metadata['class_name'] == 'Calculator'
    
    def test_parse_python_imports(self):
        """Test parsing Python imports."""
        code = """
import os
import sys
from pathlib import Path
from typing import List, Dict

def main():
    pass
"""
        
        temp_file = Path(self.temp_dir) / "imports.py"
        with open(temp_file, 'w') as f:
            f.write(code)
        
        chunks = self.parser.parse_file(temp_file)
        
        import_chunks = [c for c in chunks if c.chunk_type == 'imports']
        assert len(import_chunks) == 1
        
        import_chunk = import_chunks[0]
        assert 'import os' in import_chunk.content
        assert 'from pathlib import Path' in import_chunk.content
    
    def test_parse_javascript_function(self):
        """Test parsing a JavaScript function."""
        code = """
function greet(name) {
    return `Hello, ${name}!`;
}

const multiply = (a, b) => a * b;
"""
        
        temp_file = Path(self.temp_dir) / "test.js"
        with open(temp_file, 'w') as f:
            f.write(code)
        
        chunks = self.parser.parse_file(temp_file)
        
        assert len(chunks) >= 1
        function_chunks = [c for c in chunks if c.chunk_type == 'function']
        assert len(function_chunks) >= 1
    
    def test_parse_javascript_class(self):
        """Test parsing a JavaScript class."""
        code = """
class Person {
    constructor(name) {
        this.name = name;
    }
    
    greet() {
        return `Hello, I'm ${this.name}`;
    }
}
"""
        
        temp_file = Path(self.temp_dir) / "person.js"
        with open(temp_file, 'w') as f:
            f.write(code)
        
        chunks = self.parser.parse_file(temp_file)
        
        assert len(chunks) >= 1
        class_chunks = [c for c in chunks if c.chunk_type == 'class']
        assert len(class_chunks) == 1
        
        class_chunk = class_chunks[0]
        assert class_chunk.metadata['class_name'] == 'Person'
    
    def test_parse_java_class(self):
        """Test parsing a Java class."""
        code = """
import java.util.List;
import java.util.ArrayList;

public class Example {
    private String name;
    
    public Example(String name) {
        this.name = name;
    }
    
    public String getName() {
        return name;
    }
}
"""
        
        temp_file = Path(self.temp_dir) / "Example.java"
        with open(temp_file, 'w') as f:
            f.write(code)
        
        chunks = self.parser.parse_file(temp_file)
        
        assert len(chunks) >= 1
        import_chunks = [c for c in chunks if c.chunk_type == 'imports']
        class_chunks = [c for c in chunks if c.chunk_type == 'class']
        
        assert len(import_chunks) == 1
        assert len(class_chunks) == 1
        
        class_chunk = class_chunks[0]
        assert class_chunk.metadata['class_name'] == 'Example'
    
    def test_parse_cpp_class(self):
        """Test parsing a C++ class."""
        code = """
#include <iostream>
#include <string>

class Greeter {
private:
    std::string name;
    
public:
    Greeter(const std::string& n) : name(n) {}
    
    void greet() {
        std::cout << "Hello, " << name << "!" << std::endl;
    }
};
"""
        
        temp_file = Path(self.temp_dir) / "greeter.cpp"
        with open(temp_file, 'w') as f:
            f.write(code)
        
        chunks = self.parser.parse_file(temp_file)
        
        assert len(chunks) >= 1
        include_chunks = [c for c in chunks if c.chunk_type == 'includes']
        class_chunks = [c for c in chunks if c.chunk_type == 'class']
        
        assert len(include_chunks) == 1
        assert len(class_chunks) == 1
        
        class_chunk = class_chunks[0]
        assert class_chunk.metadata['class_name'] == 'Greeter'
    
    def test_parse_directory(self):
        """Test parsing a directory with multiple files."""
        # Create multiple files
        files = {
            "main.py": "def main():\n    print('Hello')\n",
            "utils.py": "import os\ndef helper():\n    pass\n",
            "test.js": "function test() {\n    console.log('test');\n}\n"
        }
        
        for filename, content in files.items():
            file_path = Path(self.temp_dir) / filename
            with open(file_path, 'w') as f:
                f.write(content)
        
        chunks = self.parser.parse_directory(Path(self.temp_dir))
        
        assert len(chunks) >= 3  # At least one chunk per file
        
        # Check that we have chunks from different file types
        python_chunks = [c for c in chunks if c.file_path.endswith('.py')]
        js_chunks = [c for c in chunks if c.file_path.endswith('.js')]
        
        assert len(python_chunks) >= 2
        assert len(js_chunks) >= 1
    
    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        temp_file = Path(self.temp_dir) / "test.txt"
        with open(temp_file, 'w') as f:
            f.write("This is a text file")
        
        chunks = self.parser.parse_file(temp_file)
        assert len(chunks) == 0  # Should not parse unsupported files
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent files."""
        temp_file = Path(self.temp_dir) / "nonexistent.py"
        chunks = self.parser.parse_file(temp_file)
        assert len(chunks) == 0
    
    def test_code_chunk_metadata(self):
        """Test CodeChunk metadata handling."""
        chunk = CodeChunk(
            content="def test(): pass",
            chunk_type="function",
            file_path="test.py",
            start_line=1,
            end_line=1,
            metadata={"function_name": "test"}
        )
        
        assert chunk.content == "def test(): pass"
        assert chunk.chunk_type == "function"
        assert chunk.file_path == "test.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 1
        assert chunk.metadata["function_name"] == "test"
        
        # Test to_dict method
        chunk_dict = chunk.to_dict()
        assert chunk_dict["content"] == "def test(): pass"
        assert chunk_dict["chunk_type"] == "function"
        assert chunk_dict["metadata"]["function_name"] == "test"

if __name__ == "__main__":
    pytest.main([__file__]) 