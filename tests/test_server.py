import pytest
from src.server import DeveloperWorkflowServer

SAMPLE_CODE = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""


@pytest.mark.asyncio
async def test_review_pull_request():
    server = DeveloperWorkflowServer()
    result = await server.review_pull_request(SAMPLE_CODE, "python")

    assert "static_issues" in result
    assert "ai_suggestions" in result
    assert "overall_score" in result
    print("\n✅ Review Result:", result)


@pytest.mark.asyncio
async def test_generate_documentation():
    server = DeveloperWorkflowServer()
    doc_result = await server.generate_documentation(SAMPLE_CODE, "markdown")

    assert "documentation" in doc_result
    assert "entities_documented" in doc_result
    print("\n✅ Documentation Result:", doc_result)


@pytest.mark.asyncio
async def test_generate_tests():
    server = DeveloperWorkflowServer()
    tests_result = await server.generate_tests(SAMPLE_CODE, "pytest")

    assert "test_code" in tests_result
    print("\n✅ Test Generation Result:", tests_result)


@pytest.mark.asyncio
async def test_detect_bugs():
    server = DeveloperWorkflowServer()
    bugs = await server.detect_bugs(SAMPLE_CODE, "all")

    assert "bugs_found" in bugs
    assert "bugs" in bugs
    print("\n✅ Bugs Result:", bugs)


@pytest.mark.asyncio
async def test_analyze_complexity():
    server = DeveloperWorkflowServer()
    complexity = await server.analyze_complexity(SAMPLE_CODE)

    assert "complexity_score" in complexity
    assert "metrics" in complexity
    print("\n✅ Complexity Result:", complexity)
