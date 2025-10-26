"""
Routes for Pull Request (PR) analysis and documentation generation.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from backend.services.pr_service import PRService
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/pr", tags=["Pull Requests"])

pr_service = PRService()


class PRRequest(BaseModel):
    title: str = Field(..., description="Title of the pull request")
    description: str = Field(..., description="Description or summary of the pull request")
    diff: str | None = Field(None, description="Optional diff content for deeper analysis")
    code: str | None = Field(None, description="Optional code snippet for bug/suggestion analysis")


@router.post("/analyze")
async def analyze_pr(request: PRRequest):
    """
    Analyze a Pull Request for suggestions, documentation improvements, and potential issues.

    Returns:
        - A structured JSON response with:
          - `summary`: PR overview
          - `suggestions`: action items and insights
          - `documentation`: generated doc summaries (if any)
          - `issues`: possible code-level problems
    """
    try:
        logger.info(f"🔍 Analyzing PR: {request.title}")
        result = await pr_service.analyze_pr(
            title=request.title,
            description=request.description,
            diff=request.diff,
            code=request.code
        )

        logger.info(f"✅ PR analysis completed for '{request.title}'")
        return {"status": "success", "result": result}

    except HTTPException as http_err:
        logger.error(f"❌ PR analysis failed: {http_err.detail}")
        raise http_err

    except Exception as e:
        logger.exception(f"❌ Unexpected error while analyzing PR '{request.title}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during PR analysis")
