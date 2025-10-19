# tools.py
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class Tool:
    name: str
    description: str
    inputSchema: Dict[str, Any]


def get_tools() -> List[Tool]:
    """Return all tools available to the MCP Server"""
    base_tools = [
        Tool(
            name="review_pull_request",
            description="AI-powered code review with suggestions and bug detection",
            inputSchema={
                "type": "object",
                "properties": {
                    "pr_content": {
                        "type": "string",
                        "description": "Pull request diff or code content",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (python, javascript, etc.)",
                        "default": "python",
                    },
                },
                "required": ["pr_content"],
            },
        ),
        Tool(
            name="generate_documentation",
            description="Auto-generate technical documentation from code",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_content": {
                        "type": "string",
                        "description": "Source code to document",
                    },
                    "doc_style": {
                        "type": "string",
                        "description": "Documentation style (markdown, restructuredtext, docstring)",
                        "default": "markdown",
                    },
                },
                "required": ["code_content"],
            },
        ),
        Tool(
            name="detect_bugs",
            description="Static analysis and AI-powered bug detection",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_content": {
                        "type": "string",
                        "description": "Code to analyze for bugs",
                    },
                    "severity_filter": {
                        "type": "string",
                        "description": "Filter by severity (critical, high, medium, low)",
                        "default": "all",
                    },
                },
                "required": ["code_content"],
            },
        ),
        Tool(
            name="analyze_complexity",
            description="Analyze code complexity and suggest refactoring",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_content": {
                        "type": "string",
                        "description": "Code to analyze",
                    }
                },
                "required": ["code_content"],
            },
        ),
        Tool(
            name="generate_tests",
            description="Generate unit tests for given code",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_content": {
                        "type": "string",
                        "description": "Code to generate tests for",
                    },
                    "test_framework": {
                        "type": "string",
                        "description": "Testing framework (pytest, unittest, jest)",
                        "default": "pytest",
                    },
                },
                "required": ["code_content"],
            },
        ),
    ]

    # ✅ Add contextual RAG search tool (FAISS backend)
    base_tools.append(
        Tool(
            name="contextual_search",
            description="Perform multi-vector contextual retrieval using project-aware RAG (FAISS)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or code-based search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant results to retrieve",
                        "default": 5,
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional programming language filter",
                    },
                },
                "required": ["query"],
            },
        )
    )

    # ✅ Add project indexing tool
    base_tools.append(
        Tool(
            name="index_project_context",
            description="Index project files or documentation for contextual retrieval (FAISS RAG)",
            inputSchema={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to index",
                    },
                    "root_path": {
                        "type": "string",
                        "description": "Optional repository root path to index all files",
                    },
                },
                "required": ["files"],
            },
        )
    )

    return base_tools
