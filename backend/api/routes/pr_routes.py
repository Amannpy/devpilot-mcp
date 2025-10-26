from fastapi import APIRouter, HTTPException
from backend.services.pr_service import PRService

router = APIRouter()
pr_service = PRService()

@router.post("/analyze")
async def analyze_pr(title: str, description: str, diff: str = None, code: str = None):
    """
    Analyze a Pull Request: suggestions, documentation, and potential issues.
    """
    try:
        result = await pr_service.analyze_pr(title, description, diff, code)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
