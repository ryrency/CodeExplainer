"""
LLM Agent for comprehensive repository analysis using the MCP server.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class RepoAnalysisResult:
    """Result of repository analysis."""
    repository_path: str
    architecture_summary: str
    design_patterns: List[str]
    external_dependencies: List[str]
    main_components: List[str]
    code_quality_insights: str
    recommendations: List[str]
    analysis_time: float

class MCPToolWrapper:
    """Wrapper for MCP server tools."""
    
    def __init__(self, mcp_server_process):
        self.mcp_server_process = mcp_server_process
    
    async def ask_question(self, question: str, repository_path: str) -> str:
        """Ask a question using the MCP server."""
        # This would normally communicate with the MCP server
        # For now, we'll simulate the interaction
        return f"Answer to '{question}' for repository {repository_path}"
    
    async def get_code_context(self, query: str, repository_path: str, max_results: int = 5) -> str:
        """Get code context using the MCP server."""
        return f"Code context for '{query}' in {repository_path}"

class RepoAnalyzerAgent:
    """LLM agent for comprehensive repository analysis."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize LLM
        if config.openai_api_key:
            self.llm = ChatOpenAI(
                model=config.llm_model,
                openai_api_key=config.openai_api_key,
                temperature=0.1
            )
        else:
            raise ValueError("OpenAI API key is required for the agent")
        
        # Initialize MCP tool wrapper
        self.mcp_tools = MCPToolWrapper(None)  # Will be properly initialized later
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent."""
        tools = [
            Tool(
                name="ask_code_question",
                description="Ask a specific question about the codebase using the MCP server",
                func=self._ask_code_question
            ),
            Tool(
                name="get_code_context",
                description="Get relevant code snippets for a specific query",
                func=self._get_code_context
            ),
            Tool(
                name="analyze_file_structure",
                description="Analyze the file and directory structure of the repository",
                func=self._analyze_file_structure
            ),
            Tool(
                name="extract_dependencies",
                description="Extract external dependencies from configuration files",
                func=self._extract_dependencies
            ),
            Tool(
                name="identify_patterns",
                description="Identify common design patterns in the codebase",
                func=self._identify_patterns
            )
        ]
        return tools
    
    def _create_agent(self):
        """Create the LLM agent."""
        system_prompt = """You are an expert software architect and code analyst. Your task is to analyze a repository and provide comprehensive insights about its architecture, design patterns, dependencies, and code quality.

Your analysis should cover:
1. Overall architecture and structure
2. Design patterns used
3. External dependencies and libraries
4. Main components and their responsibilities
5. Code quality insights
6. Recommendations for improvement

Use the available tools to gather information about the codebase. Be thorough and provide specific examples from the code when possible."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        return create_openai_tools_agent(self.llm, self.tools, prompt)
    
    async def _ask_code_question(self, question: str, repository_path: str) -> str:
        """Ask a question about the codebase."""
        try:
            return await self.mcp_tools.ask_question(question, repository_path)
        except Exception as e:
            logger.error(f"Error asking code question: {e}")
            return f"Error: {str(e)}"
    
    async def _get_code_context(self, query: str, repository_path: str, max_results: int = 5) -> str:
        """Get code context for a query."""
        try:
            return await self.mcp_tools.get_code_context(query, repository_path, max_results)
        except Exception as e:
            logger.error(f"Error getting code context: {e}")
            return f"Error: {str(e)}"
    
    def _analyze_file_structure(self, repository_path: str) -> str:
        """Analyze the file structure of the repository."""
        try:
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return f"Repository path does not exist: {repository_path}"
            
            # Analyze directory structure
            structure = []
            for item in repo_path.rglob('*'):
                if item.is_file():
                    rel_path = item.relative_to(repo_path)
                    structure.append(f"File: {rel_path}")
                elif item.is_dir() and item != repo_path:
                    rel_path = item.relative_to(repo_path)
                    structure.append(f"Directory: {rel_path}/")
            
            # Count file types
            file_extensions = {}
            for item in repo_path.rglob('*'):
                if item.is_file():
                    ext = item.suffix
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
            
            analysis = f"Repository Structure Analysis for {repository_path}:\n\n"
            analysis += f"Total files: {len([f for f in repo_path.rglob('*') if f.is_file()])}\n"
            analysis += f"Total directories: {len([d for d in repo_path.rglob('*') if d.is_dir()])}\n\n"
            
            analysis += "File types:\n"
            for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True):
                analysis += f"  {ext or 'no extension'}: {count} files\n"
            
            analysis += "\nTop-level structure:\n"
            for item in sorted(repo_path.iterdir()):
                if item.is_dir():
                    analysis += f"  📁 {item.name}/\n"
                else:
                    analysis += f"  📄 {item.name}\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing file structure: {e}")
            return f"Error analyzing file structure: {str(e)}"
    
    def _extract_dependencies(self, repository_path: str) -> str:
        """Extract external dependencies from configuration files."""
        try:
            repo_path = Path(repository_path)
            dependencies = {}
            
            # Check for common dependency files
            dependency_files = {
                'requirements.txt': 'Python',
                'pyproject.toml': 'Python',
                'package.json': 'Node.js',
                'pom.xml': 'Java/Maven',
                'build.gradle': 'Java/Gradle',
                'Cargo.toml': 'Rust',
                'go.mod': 'Go',
                'Gemfile': 'Ruby',
                'composer.json': 'PHP'
            }
            
            for filename, language in dependency_files.items():
                file_path = repo_path / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        if filename == 'requirements.txt':
                            deps = [line.strip().split('==')[0].split('>=')[0].split('<=')[0] 
                                   for line in content.split('\n') 
                                   if line.strip() and not line.startswith('#')]
                        elif filename == 'package.json':
                            import json
                            data = json.loads(content)
                            deps = list(data.get('dependencies', {}).keys()) + list(data.get('devDependencies', {}).keys())
                        else:
                            deps = [f"Found {filename} but parsing not implemented"]
                        
                        dependencies[language] = deps
                        
                    except Exception as e:
                        dependencies[language] = [f"Error parsing {filename}: {str(e)}"]
            
            if not dependencies:
                return "No standard dependency files found in the repository."
            
            analysis = "External Dependencies Analysis:\n\n"
            for language, deps in dependencies.items():
                analysis += f"{language} Dependencies:\n"
                for dep in deps[:10]:  # Limit to first 10
                    analysis += f"  - {dep}\n"
                if len(deps) > 10:
                    analysis += f"  ... and {len(deps) - 10} more\n"
                analysis += "\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error extracting dependencies: {e}")
            return f"Error extracting dependencies: {str(e)}"
    
    def _identify_patterns(self, repository_path: str) -> str:
        """Identify common design patterns in the codebase."""
        try:
            # This would normally analyze the code structure
            # For now, we'll provide a template analysis
            patterns = [
                "MVC (Model-View-Controller) - Common in web applications",
                "Repository Pattern - For data access abstraction",
                "Factory Pattern - For object creation",
                "Observer Pattern - For event handling",
                "Singleton Pattern - For global state management"
            ]
            
            analysis = "Design Patterns Analysis:\n\n"
            analysis += "Common patterns that might be present:\n"
            for pattern in patterns:
                analysis += f"  - {pattern}\n"
            
            analysis += "\nNote: This is a general analysis. Use the code question tool to identify specific patterns in the codebase."
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error identifying patterns: {e}")
            return f"Error identifying patterns: {str(e)}"
    
    async def analyze_repository(self, repository_path: str) -> RepoAnalysisResult:
        """Perform comprehensive repository analysis."""
        import time
        start_time = time.time()
        
        logger.info(f"Starting analysis of repository: {repository_path}")
        
        try:
            # Create analysis prompt
            analysis_prompt = f"""
Please analyze the repository at {repository_path} and provide a comprehensive report covering:

1. **Architecture Summary**: Describe the overall architecture and structure of the codebase
2. **Design Patterns**: Identify the main design patterns used in the code
3. **External Dependencies**: List the key external libraries and frameworks used
4. **Main Components**: Describe the main components and their responsibilities
5. **Code Quality Insights**: Provide insights about code quality, organization, and best practices
6. **Recommendations**: Suggest improvements or areas for enhancement

Use the available tools to gather detailed information about the codebase. Be specific and provide examples from the code when possible.
"""

            # Run the agent
            result = await self.agent_executor.ainvoke({
                "input": analysis_prompt,
                "chat_history": []
            })
            
            # Parse the result and extract structured information
            analysis_text = result.get("output", "")
            
            # Extract structured information from the analysis
            architecture_summary = self._extract_section(analysis_text, "architecture", "design patterns")
            design_patterns = self._extract_list_items(analysis_text, "design patterns", "external dependencies")
            external_dependencies = self._extract_list_items(analysis_text, "external dependencies", "main components")
            main_components = self._extract_list_items(analysis_text, "main components", "code quality")
            code_quality_insights = self._extract_section(analysis_text, "code quality", "recommendations")
            recommendations = self._extract_list_items(analysis_text, "recommendations", "")
            
            analysis_time = time.time() - start_time
            
            return RepoAnalysisResult(
                repository_path=repository_path,
                architecture_summary=architecture_summary,
                design_patterns=design_patterns,
                external_dependencies=external_dependencies,
                main_components=main_components,
                code_quality_insights=code_quality_insights,
                recommendations=recommendations,
                analysis_time=analysis_time
            )
            
        except Exception as e:
            logger.error(f"Error analyzing repository: {e}")
            analysis_time = time.time() - start_time
            
            return RepoAnalysisResult(
                repository_path=repository_path,
                architecture_summary=f"Error during analysis: {str(e)}",
                design_patterns=[],
                external_dependencies=[],
                main_components=[],
                code_quality_insights="",
                recommendations=[],
                analysis_time=analysis_time
            )
    
    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extract a section of text between markers."""
        try:
            start_idx = text.lower().find(start_marker.lower())
            if start_idx == -1:
                return ""
            
            if end_marker:
                end_idx = text.lower().find(end_marker.lower(), start_idx)
                if end_idx == -1:
                    end_idx = len(text)
            else:
                end_idx = len(text)
            
            return text[start_idx:end_idx].strip()
        except:
            return ""
    
    def _extract_list_items(self, text: str, start_marker: str, end_marker: str) -> List[str]:
        """Extract list items from a section of text."""
        section = self._extract_section(text, start_marker, end_marker)
        if not section:
            return []
        
        items = []
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                item = line[1:].strip()
                if item:
                    items.append(item)
        
        return items
    
    def generate_analysis_report(self, result: RepoAnalysisResult) -> str:
        """Generate a formatted analysis report."""
        report = []
        report.append("=" * 80)
        report.append("REPOSITORY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Repository: {result.repository_path}")
        report.append(f"Analysis Time: {result.analysis_time:.2f} seconds")
        report.append("")
        
        # Architecture Summary
        report.append("🏗️  ARCHITECTURE SUMMARY")
        report.append("-" * 40)
        report.append(result.architecture_summary)
        report.append("")
        
        # Design Patterns
        report.append("🎯 DESIGN PATTERNS")
        report.append("-" * 40)
        if result.design_patterns:
            for pattern in result.design_patterns:
                report.append(f"• {pattern}")
        else:
            report.append("No specific design patterns identified.")
        report.append("")
        
        # External Dependencies
        report.append("📦 EXTERNAL DEPENDENCIES")
        report.append("-" * 40)
        if result.external_dependencies:
            for dep in result.external_dependencies:
                report.append(f"• {dep}")
        else:
            report.append("No external dependencies identified.")
        report.append("")
        
        # Main Components
        report.append("🔧 MAIN COMPONENTS")
        report.append("-" * 40)
        if result.main_components:
            for component in result.main_components:
                report.append(f"• {component}")
        else:
            report.append("No main components identified.")
        report.append("")
        
        # Code Quality Insights
        report.append("📊 CODE QUALITY INSIGHTS")
        report.append("-" * 40)
        report.append(result.code_quality_insights)
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 40)
        if result.recommendations:
            for rec in result.recommendations:
                report.append(f"• {rec}")
        else:
            report.append("No specific recommendations provided.")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main function for repository analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze repository using LLM agent')
    parser.add_argument('--repository', '-r', required=True,
                       help='Path to the repository to analyze')
    parser.add_argument('--output', '-o', default='data/analysis/report.txt',
                       help='Output path for analysis report')
    parser.add_argument('--json', default='data/analysis/results.json',
                       help='Output path for JSON results')
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = Config()
        config.validate()
        
        # Initialize agent
        agent = RepoAnalyzerAgent(config)
        
        # Check if repository exists
        repo_path = Path(args.repository)
        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {args.repository}")
            sys.exit(1)
        
        # Run analysis
        logger.info(f"Starting repository analysis: {args.repository}")
        result = await agent.analyze_repository(args.repository)
        
        # Generate report
        report = agent.generate_analysis_report(result)
        
        # Save report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        # Save JSON results
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        json_data = {
            'repository_path': result.repository_path,
            'architecture_summary': result.architecture_summary,
            'design_patterns': result.design_patterns,
            'external_dependencies': result.external_dependencies,
            'main_components': result.main_components,
            'code_quality_insights': result.code_quality_insights,
            'recommendations': result.recommendations,
            'analysis_time': result.analysis_time
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED")
        print("="*60)
        print(f"Repository: {result.repository_path}")
        print(f"Analysis Time: {result.analysis_time:.2f} seconds")
        print(f"Design Patterns Found: {len(result.design_patterns)}")
        print(f"Dependencies Found: {len(result.external_dependencies)}")
        print(f"Components Identified: {len(result.main_components)}")
        print(f"Recommendations: {len(result.recommendations)}")
        print(f"\nReport saved to: {args.output}")
        print(f"JSON results saved to: {args.json}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 