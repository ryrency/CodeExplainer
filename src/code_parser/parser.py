"""
Code parser for intelligent chunking of source code into logical blocks.
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re

logger = logging.getLogger(__name__)

class CodeChunk:
    """Represents a logical chunk of code."""
    
    def __init__(self, 
                 content: str, 
                 chunk_type: str, 
                 file_path: str, 
                 start_line: int, 
                 end_line: int,
                 metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.chunk_type = chunk_type  # 'function', 'class', 'method', 'module', 'import'
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"{self.chunk_type} in {self.file_path}:{self.start_line}-{self.end_line}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            'content': self.content,
            'chunk_type': self.chunk_type,
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'metadata': self.metadata
        }

class CodeParser:
    """Intelligent code parser that chunks code into logical blocks."""
    
    def __init__(self):
        self.supported_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp'}
    
    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single file and return logical code chunks."""
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return []
        
        if file_path.suffix not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = []
            if file_path.suffix == '.py':
                chunks = self._parse_python_file(file_path, content)
            elif file_path.suffix in {'.js', '.ts'}:
                chunks = self._parse_javascript_file(file_path, content)
            elif file_path.suffix in {'.java'}:
                chunks = self._parse_java_file(file_path, content)
            elif file_path.suffix in {'.cpp', '.c', '.h', '.hpp'}:
                chunks = self._parse_cpp_file(file_path, content)
            else:
                chunks = self._parse_generic_file(file_path, content)
            
            # Validate that all chunks are CodeChunk objects
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, CodeChunk):
                    valid_chunks.append(chunk)
                else:
                    logger.warning(f"Invalid chunk type in {file_path}: {type(chunk)} at index {i}")
            
            return valid_chunks
                
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return []
    
    def _parse_python_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Parse Python file using AST."""
        chunks = []
        
        try:
            tree = ast.parse(content)
            
            # Add module-level imports
            imports = self._extract_python_imports(content)
            if imports:
                import_lines = [l for l in imports.split('\n') if l.strip()]
                chunks.append(CodeChunk(
                    content=imports,
                    chunk_type='imports',
                    file_path=str(file_path),
                    start_line=1,
                    end_line=len(import_lines),
                    metadata={'import_count': len(import_lines)}
                ))
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    chunk = self._extract_python_class(node, content, file_path)
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.FunctionDef) and not hasattr(node, 'parent_class'):
                    chunk = self._extract_python_function(node, content, file_path)
                    if chunk:
                        chunks.append(chunk)
            
            # If no structured chunks found, create module-level chunk
            if not chunks or (len(chunks) == 1 and chunks[0].chunk_type == 'imports'):
                module_content = self._extract_module_content(content, tree)
                if module_content.strip():
                    chunks.append(CodeChunk(
                        content=module_content,
                        chunk_type='module',
                        file_path=str(file_path),
                        start_line=1,
                        end_line=len(content.split('\n')),
                        metadata={'module_level': True}
                    ))
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {str(e)}")
            # Fallback to generic parsing
            return self._parse_generic_file(file_path, content)
        
        return chunks
    
    def _extract_python_imports(self, content: str) -> str:
        """Extract import statements from Python code."""
        import_lines = []
        lines = content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') and ' import ' in stripped):
                import_lines.append(line)
            elif stripped and not stripped.startswith('#'):
                break
        
        return '\n'.join(import_lines)
    
    def _extract_python_class(self, node: ast.ClassDef, content: str, file_path: Path) -> Optional[CodeChunk]:
        """Extract a Python class definition."""
        lines = content.split('\n')
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        # Find the actual end of the class
        class_content = '\n'.join(lines[start_line - 1:end_line])
        
        # Extract methods
        methods = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method_chunk = self._extract_python_function(child, content, file_path, parent_class=node.name)
                if method_chunk:
                    methods.append(method_chunk)
        
        metadata = {
            'class_name': node.name,
            'base_classes': [base.id for base in node.bases if isinstance(base, ast.Name)],
            'method_count': len(methods),
            'methods': [m.metadata.get('function_name') for m in methods]
        }
        
        return CodeChunk(
            content=class_content,
            chunk_type='class',
            file_path=str(file_path),
            start_line=start_line,
            end_line=end_line,
            metadata=metadata
        )
    
    def _extract_python_function(self, node: ast.FunctionDef, content: str, file_path: Path, parent_class: Optional[str] = None) -> Optional[CodeChunk]:
        """Extract a Python function definition."""
        lines = content.split('\n')
        start_line = node.lineno
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        function_content = '\n'.join(lines[start_line - 1:end_line])
        
        # Extract function signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        metadata = {
            'function_name': node.name,
            'parent_class': parent_class,
            'arguments': args,
            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
        }
        
        chunk_type = 'method' if parent_class else 'function'
        
        return CodeChunk(
            content=function_content,
            chunk_type=chunk_type,
            file_path=str(file_path),
            start_line=start_line,
            end_line=end_line,
            metadata=metadata
        )
    
    def _extract_module_content(self, content: str, tree: ast.AST) -> str:
        """Extract module-level content (excluding classes and functions)."""
        lines = content.split('\n')
        module_lines = []
        
        for i, line in enumerate(lines, 1):
            # Check if this line is part of a class or function definition
            is_inside_structure = False
            for node in ast.walk(tree):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and 
                        node.lineno <= i <= node.end_lineno):
                        is_inside_structure = True
                        break
            
            if not is_inside_structure and line.strip():
                module_lines.append(line)
        
        return '\n'.join(module_lines)
    
    def _parse_javascript_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Parse JavaScript/TypeScript file using regex patterns."""
        chunks = []
        
        # Extract imports
        import_pattern = r'^(import\s+.*?;?\s*$|export\s+.*?;?\s*$)'
        imports = re.findall(import_pattern, content, re.MULTILINE)
        if imports:
            import_content = '\n'.join(imports)
            chunks.append(CodeChunk(
                content=import_content,
                chunk_type='imports',
                file_path=str(file_path),
                start_line=1,
                end_line=len(imports),
                metadata={'import_count': len(imports)}
            ))
        
        # Extract classes
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{'
        class_matches = list(re.finditer(class_pattern, content))
        
        for match in class_matches:
            class_name = match.group(1)
            base_class = match.group(2)
            
            # Find class end
            start_pos = match.start()
            brace_count = 0
            end_pos = start_pos
            
            for i, char in enumerate(content[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            class_content = content[start_pos:end_pos]
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            metadata = {
                'class_name': class_name,
                'base_class': base_class
            }
            
            chunks.append(CodeChunk(
                content=class_content,
                chunk_type='class',
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line,
                metadata=metadata
            ))
        
        # Extract functions
        function_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{'
        function_matches = list(re.finditer(function_pattern, content))
        
        for match in function_matches:
            function_name = match.group(1)
            start_pos = match.start()
            
            # Find function end
            brace_count = 0
            end_pos = start_pos
            
            for i, char in enumerate(content[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            function_content = content[start_pos:end_pos]
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            metadata = {
                'function_name': function_name
            }
            
            chunks.append(CodeChunk(
                content=function_content,
                chunk_type='function',
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line,
                metadata=metadata
            ))
        
        return chunks
    
    def _parse_java_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Parse Java file using regex patterns."""
        chunks = []
        
        # Extract imports
        import_pattern = r'^import\s+.*?;'
        imports = re.findall(import_pattern, content, re.MULTILINE)
        if imports:
            import_content = '\n'.join(imports)
            chunks.append(CodeChunk(
                content=import_content,
                chunk_type='imports',
                file_path=str(file_path),
                start_line=1,
                end_line=len(imports),
                metadata={'import_count': len(imports)}
            ))
        
        # Extract classes
        class_pattern = r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{'
        class_matches = list(re.finditer(class_pattern, content))
        
        for match in class_matches:
            class_name = match.group(1)
            base_class = match.group(2)
            interfaces = match.group(3)
            
            # Find class end
            start_pos = match.start()
            brace_count = 0
            end_pos = start_pos
            
            for i, char in enumerate(content[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            class_content = content[start_pos:end_pos]
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            metadata = {
                'class_name': class_name,
                'base_class': base_class,
                'interfaces': interfaces.split(',') if interfaces else []
            }
            
            chunks.append(CodeChunk(
                content=class_content,
                chunk_type='class',
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line,
                metadata=metadata
            ))
        
        return chunks
    
    def _parse_cpp_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Parse C++ file using regex patterns."""
        chunks = []
        
        # Extract includes
        include_pattern = r'^#include\s+[<"][^>"]*[>"]'
        includes = re.findall(include_pattern, content, re.MULTILINE)
        if includes:
            include_content = '\n'.join(includes)
            chunks.append(CodeChunk(
                content=include_content,
                chunk_type='includes',
                file_path=str(file_path),
                start_line=1,
                end_line=len(includes),
                metadata={'include_count': len(includes)}
            ))
        
        # Extract classes
        class_pattern = r'class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+(\w+))?\s*\{'
        class_matches = list(re.finditer(class_pattern, content))
        
        for match in class_matches:
            class_name = match.group(1)
            base_class = match.group(2)
            
            # Find class end
            start_pos = match.start()
            brace_count = 0
            end_pos = start_pos
            
            for i, char in enumerate(content[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            class_content = content[start_pos:end_pos]
            start_line = content[:start_pos].count('\n') + 1
            end_line = content[:end_pos].count('\n') + 1
            
            metadata = {
                'class_name': class_name,
                'base_class': base_class
            }
            
            chunks.append(CodeChunk(
                content=class_content,
                chunk_type='class',
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line,
                metadata=metadata
            ))
        
        return chunks
    
    def _parse_generic_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Generic file parser for unsupported file types."""
        lines = content.split('\n')
        
        # Create a single chunk for the entire file
        return [CodeChunk(
            content=content,
            chunk_type='file',
            file_path=str(file_path),
            start_line=1,
            end_line=len(lines),
            metadata={'file_type': file_path.suffix}
        )]
    
    def parse_directory(self, directory_path: Path) -> List[CodeChunk]:
        """Parse all supported files in a directory recursively."""
        chunks = []
        
        # Directories to exclude
        exclude_dirs = {'venv', '__pycache__', '.git', 'node_modules', '.pytest_cache', '.mypy_cache'}
        
        for file_path in directory_path.rglob('*'):
            # Skip excluded directories
            if any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                continue
                
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                try:
                    file_chunks = self.parse_file(file_path)
                    chunks.extend(file_chunks)
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")
                    continue
        
        return chunks 