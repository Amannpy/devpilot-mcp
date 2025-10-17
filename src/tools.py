# tools.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Tool:
    name: str
    description: str
    inputSchema: Dict[str, Any]

def get_tools():
    """Return all tools available to MCP Server"""
    return [
        Tool(
            name="review_pull_request",
            description="AI-powered code review with suggestions and bug detection",
            inputSchema={
                "pr_content": {"type": "string", "description": "Pull request diff or code content"},
                "language": {"type": "string", "description": "Programming language (python, javascript, etc.)", "default": "python"}
            }
        ),
        Tool(
            name="generate_documentation",
            description="Auto-generate technical documentation from code",
            inputSchema={
                "code_content": {"type": "string", "description": "Source code to document"},
                "doc_style": {"type": "string", "description": "Documentation style (markdown, restructuredtext, docstring)", "default": "markdown"}
            }
        ),
        Tool(
            name="detect_bugs",
            description="Static analysis and AI-powered bug detection",
            inputSchema={
                "code_content": {"type": "string", "description": "Code to analyze for bugs"},
                "severity_filter": {"type": "string", "description": "Filter by severity (critical, high, medium, low)", "default": "all"}
            }
        ),
        Tool(
            name="analyze_complexity",
            description="Analyze code complexity and suggest refactoring",
            inputSchema={
                "code_content": {"type": "string", "description": "Code to analyze"}
            }
        ),
        Tool(
            name="generate_tests",
            description="Generate unit tests for given code",
            inputSchema={
                "code_content": {"type": "string", "description": "Code to generate tests for"},
                "test_framework": {"type": "string", "description": "Testing framework (pytest, unittest, jest)", "default": "pytest"}
            }
        )
    ]
