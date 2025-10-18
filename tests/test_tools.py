# tests/test_tools.py
import pytest
from src.tools import get_tools


def test_tools_structure():
    tools = get_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0

    for tool in tools:
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "inputSchema")
        assert isinstance(tool.inputSchema, dict)


def test_defaults_present():
    tools = get_tools()
    for tool in tools:
        if tool.name == "review_pull_request":
            assert "language" in tool.inputSchema
            assert tool.inputSchema["language"]["default"] == "python"
        if tool.name == "generate_documentation":
            assert "doc_style" in tool.inputSchema
            assert tool.inputSchema["doc_style"]["default"] == "markdown"
        if tool.name == "generate_tests":
            assert "test_framework" in tool.inputSchema
            assert tool.inputSchema["test_framework"]["default"] == "pytest"
