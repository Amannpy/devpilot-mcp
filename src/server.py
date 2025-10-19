"""
Intelligent Developer Workflow MCP Server
AI-powered code review, documentation generation, and project management
"""

import asyncio
import json
from typing import Any
from dataclasses import dataclass
from datetime import datetime
import logging
import re

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource, Prompt
from pydantic import AnyUrl

from src.models import ModelManager  # <-- use Qwen2.5 integration
from src.tools import get_tools
from src.rag import RAGManager

# create a single rag manager instance
rag_manager = RAGManager()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeAnalysisResult:
    """Structure for code analysis results"""

    issues: list[dict]
    suggestions: list[str]
    complexity_score: float
    documentation_gaps: list[str]


class DeveloperWorkflowServer:
    """Main MCP server for developer workflows"""

    def __init__(self):
        self.server = Server("devpilot-mcp")
        self.model_manager = ModelManager()  # Qwen2.5 manager
        self.setup_handlers()

        # Cache for model responses
        self.response_cache = {}

    def setup_handlers(self):
        """Register all MCP handlers"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available development tools"""
            tools = get_tools()

            # Add RAG-powered contextual retrieval tool
            tools.append(
                Tool(
                    name="contextual_search",
                    description="Perform multi-vector contextual retrieval using project-aware RAG",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Natural language or code-based search query"},
                            "top_k": {"type": "integer", "description": "Number of relevant results to retrieve",
                                      "default": 5},
                        },
                        "required": ["query"],
                    },
                )
            )

            return get_tools()
            # return [
            #     Tool(
            #         name="review_pull_request",
            #         description="AI-powered code review with suggestions and bug detection",
            #         inputSchema={
            #             "type": "object",
            #             "properties": {
            #                 "pr_content": {"type": "string", "description": "Pull request diff or code content"},
            #                 "language": {"type": "string", "description": "Programming language", "default": "python"}
            #             },
            #             "required": ["pr_content"]
            #         }
            #     ),
            #     Tool(
            #         name="generate_documentation",
            #         description="Auto-generate technical documentation from code",
            #         inputSchema={
            #             "type": "object",
            #             "properties": {
            #                 "code_content": {"type": "string", "description": "Source code to document"},
            #                 "doc_style": {"type": "string", "description": "Documentation style", "default": "markdown"}
            #             },
            #             "required": ["code_content"]
            #         }
            #     ),
            #     Tool(
            #         name="detect_bugs",
            #         description="Static analysis and AI-powered bug detection",
            #         inputSchema={
            #             "type": "object",
            #             "properties": {
            #                 "code_content": {"type": "string", "description": "Code to analyze for bugs"},
            #                 "severity_filter": {"type": "string", "description": "Filter by severity", "default": "all"}
            #             },
            #             "required": ["code_content"]
            #         }
            #     ),
            #     Tool(
            #         name="analyze_complexity",
            #         description="Analyze code complexity and suggest refactoring",
            #         inputSchema={
            #             "type": "object",
            #             "properties": {
            #                 "code_content": {"type": "string", "description": "Code to analyze"}
            #             },
            #             "required": ["code_content"]
            #         }
            #     ),
            #     Tool(
            #         name="generate_tests",
            #         description="Generate unit tests for given code",
            #         inputSchema={
            #             "type": "object",
            #             "properties": {
            #                 "code_content": {"type": "string", "description": "Code to generate tests for"},
            #                 "test_framework": {"type": "string", "description": "Testing framework", "default": "pytest"}
            #             },
            #             "required": ["code_content"]
            #         }
            #     )
            # ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Handle tool execution"""
            try:
                if name == "review_pull_request":
                    result = await self.review_pull_request(
                        arguments["pr_content"], arguments.get("language", "python")
                    )
                elif name == "generate_documentation":
                    result = await self.generate_documentation(
                        arguments["code_content"], arguments.get("doc_style", "markdown")
                    )
                elif name == "detect_bugs":
                    result = await self.detect_bugs(
                        arguments["code_content"], arguments.get("severity_filter", "all")
                    )
                elif name == "analyze_complexity":
                    result = await self.analyze_complexity(arguments["code_content"])
                elif name == "generate_tests":
                    result = await self.generate_tests(
                        arguments["code_content"], arguments.get("test_framework", "pytest")
                    )
                elif name == "contextual_search":
                    from src.rag import rag_manager
                    query = arguments.get("query", "")
                    top_k = int(arguments.get("top_k", 5))
                    results = await rag_manager.query(query, top_k=top_k)
                    result = {"query": query, "top_k": top_k, "results": results}
                else:
                    raise ValueError(f"Unknown tool: {name}")

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            """List available resources"""
            return [
                Resource(
                    uri=AnyUrl("git://repository/analysis"),
                    name="Repository Analysis",
                    description="Analyze git repository structure and health",
                    mimeType="application/json",
                ),
                Resource(
                    uri=AnyUrl("project://metrics/overview"),
                    name="Project Metrics",
                    description="Code quality metrics and statistics",
                    mimeType="application/json",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource content"""
            if uri == "git://repository/analysis":
                return json.dumps(
                    {
                        "status": "healthy",
                        "metrics": {
                            "total_files": 150,
                            "code_coverage": 85.5,
                            "tech_debt_hours": 12.3,
                        },
                    }
                )
            elif uri == "project://metrics/overview":
                return json.dumps(
                    {"code_quality": "A", "maintainability_index": 78, "test_coverage": 85.5}
                )
            return "{}"

        @self.server.list_prompts()
        async def list_prompts() -> list[Prompt]:
            """List available prompts"""
            return [
                Prompt(
                    name="code-review-template",
                    description="Template for code review comments",
                    arguments=[],
                )
            ]

    # -----------------------------
    # Tool implementations using Qwen2.5
    # -----------------------------
    async def review_pull_request(self, pr_content: str, language: str) -> dict:
        """AI-powered code review using Qwen2.5"""
        cache_key = f"review_{hash(pr_content)}"
        if cache_key in self.response_cache:
            logger.info("Returning cached review")
            return self.response_cache[cache_key]

        # Static analysis
        issues = self._static_analysis(pr_content, language)

        # AI-powered review
        try:
            ai_suggestions = await self.model_manager.generate_review(pr_content, language)
        except Exception as e:
            logger.warning(f"AI review failed: {e}, using static analysis only")
            ai_suggestions = []

        result = {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "static_issues": issues,
            "ai_suggestions": ai_suggestions,
            "overall_score": self._calculate_score(issues, ai_suggestions),
            "summary": self._generate_summary(issues, ai_suggestions),
        }

        # Cache result
        self.response_cache[cache_key] = result
        return result

    async def generate_documentation(self, code_content: str, doc_style: str) -> dict:
        """Generate documentation using Qwen2.5"""
        try:
            documentation = await self.model_manager.generate_documentation(code_content)
            entities = self._extract_entities(code_content)

            return {
                "documentation": documentation,
                "style": doc_style,
                "entities_documented": entities,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return {"error": str(e)}

    async def generate_tests(self, code_content: str, test_framework: str) -> dict:
        """Generate unit tests using Qwen2.5"""
        try:
            test_code = await self.model_manager.generate_tests(code_content)
            return {
                "test_code": test_code,
                "framework": test_framework,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": str(e)}

    async def detect_bugs(self, code_content: str, severity_filter: str) -> dict:
        """Detect bugs using pattern matching"""
        bugs = []
        patterns = {
            "sql_injection": (r"execute\(.*\+.*\)", "critical"),
            "hardcoded_password": (r"password\s*=\s*['\"].*['\"]", "high"),
            "eval_usage": (r"eval\(", "high"),
            "bare_except": (r"except\s*:", "medium"),
            "print_debug": (r"print\(", "low"),
        }

        for bug_type, (pattern, severity) in patterns.items():
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                if severity_filter == "all" or severity_filter == severity:
                    bugs.append(
                        {
                            "type": bug_type,
                            "severity": severity,
                            "line": code_content[: match.start()].count("\n") + 1,
                            "snippet": match.group(0),
                        }
                    )

        return {
            "bugs_found": len(bugs),
            "bugs": bugs,
            "severity_filter": severity_filter,
            "analyzed_at": datetime.now().isoformat(),
        }

    async def analyze_complexity(self, code_content: str) -> dict:
        """Analyze code complexity"""
        lines = code_content.split("\n")
        functions = len(re.findall(r"def\s+\w+", code_content))
        classes = len(re.findall(r"class\s+\w+", code_content))
        nesting_level = self._calculate_max_nesting(code_content)

        complexity_score = float(
            min(100.0, (len(lines) * 0.1 + functions * 2 + classes * 3 + nesting_level * 5))
        )

        return {
            "complexity_score": round(complexity_score, 2),
            "metrics": {
                "lines_of_code": len(lines),
                "functions": functions,
                "classes": classes,
                "max_nesting_level": nesting_level,
            },
            "recommendations": self._generate_refactoring_suggestions(complexity_score),
        }

    # -----------------------------
    # Helper methods (static analysis, score, etc.)
    # -----------------------------
    def _static_analysis(self, code: str, language: str) -> list[dict]:
        """Simple static analysis"""
        issues = []
        if "TODO" in code or "FIXME" in code:
            issues.append(
                {
                    "type": "todo_comment",
                    "severity": "low",
                    "message": "Contains TODO/FIXME comments",
                }
            )
        if len(code.split("\n")) > 500:
            issues.append(
                {
                    "type": "large_file",
                    "severity": "medium",
                    "message": "File is very large, consider splitting",
                }
            )
        return issues

    def _calculate_score(self, issues: list, suggestions: list) -> float:
        base_score = 100.0
        base_score -= len(issues) * 5
        base_score -= len(suggestions) * 3
        return max(0, min(100, base_score))

    def _generate_summary(self, issues: list, suggestions: list) -> str:
        if not issues and not suggestions:
            return "Code looks good! No major issues found."
        return f"Found {len(issues)} static issues and {len(suggestions)} suggestions."

    def _extract_entities(self, code: str) -> list[str]:
        entities = []
        entities.extend(re.findall(r"def\s+(\w+)", code))
        entities.extend(re.findall(r"class\s+(\w+)", code))
        return entities

    def _calculate_max_nesting(self, code: str) -> int:
        max_level = 0
        for line in code.split("\n"):
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                level = indent // 4
                max_level = max(max_level, level)
        return max_level

    def _generate_refactoring_suggestions(self, score: float) -> list[str]:
        # Explicit type conversion for mypy
        score_value: float = float(score)
        if score_value < 30:
            return ["Code complexity is acceptable"]
        elif score_value < 60:
            return ["Consider breaking down large functions", "Review nesting levels"]
        else:
            return [
                "High complexity detected",
                "Refactor into smaller modules",
                "Extract complex logic into separate functions",
            ]

    async def run(self):
        """Start the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Intelligent Developer Workflow MCP Server started")
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


async def main():
    """Entry point"""
    server = DeveloperWorkflowServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
