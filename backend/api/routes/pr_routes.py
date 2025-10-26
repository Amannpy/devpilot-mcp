from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.pr_service import analyze_pr

router = APIRouter(prefix="/pr", tags=["Pull Request Analysis"])


class PRRequest(BaseModel):
    title: str
    description: str
    diff: str | None = None  # optional: git diff or code snippet


@router.post("/analyze")
async def analyze_pull_request(payload: PRRequest):
    """
    Analyze a pull request description and code diff.
    Returns documentation insights, code suggestions, or potential bugs.
    """
    try:
        result = analyze_pr(payload.title, payload.description, payload.diff)
        return {"status": "success", "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
