import pytest
from backend.services.pr_service import PRService

@pytest.mark.asyncio
async def test_pr_heuristics_refactor():
    pr_service = PRService()
    result = await pr_service.analyze_pr(
        title="Refactor DB Logic",
        description="Refactored DB connection and schema handling.",
        diff="def connect():\n    pass\n" * 10
    )
    assert "heuristic" in result
    assert any("refactor" in s.lower() for s in result["heuristic"]["suggestions"])
