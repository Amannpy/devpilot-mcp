# tests/test_server.py
import pytest
from src.server import DeveloperWorkflowServer

@pytest.mark.asyncio
async def test_analyze_complexity(server):
    code_snippet = """
def foo(x):
    if x > 5:
        print("hi")
    return x
"""
    result = await server.analyze_complexity(code_snippet)
    assert "complexity_score" in result
    assert result["metrics"]["functions"] == 1

@pytest.mark.asyncio
async def test_detect_bugs(server):
    buggy_code = 'password = "1234"\nprint("debug")'
    result = await server.detect_bugs(buggy_code, 'all')
    assert result["bugs_found"] >= 1

@pytest.mark.asyncio
async def test_generate_documentation(server):
    server.hf_client.text_generation = lambda *args, **kwargs: "Mocked documentation output"
    result = await server.generate_documentation("def add(a, b): return a + b", "markdown")
    assert "documentation" in result
    assert "add" in result["entities_documented"]

@pytest.mark.asyncio
async def test_generate_tests(server):
    server.hf_client.text_generation = lambda *args, **kwargs: "Mocked test output"
    result = await server.generate_tests("def add(a, b): return a + b", "pytest")
    assert "test_code" in result
