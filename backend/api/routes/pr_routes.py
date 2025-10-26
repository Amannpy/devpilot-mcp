"""
Pull Request (PR) analysis routes for the MCP AI Server.
Handles intelligent PR reviews and AI-generated code insights.
"""

from fastapi import APIRouter, Depends, HTTPException
from backend.core.mcp_server import MCPServer

router = APIRouter()


@router.post("/analyze")
async def analyze_pr(
    title: str,
    description: str,
    diff: str | None = None,
    code: str | None = None,
    mcp_server: MCPServer = Depends(),
):
    """
    Analyze a Pull Request (PR) using the MCPServer.
    Returns heuristic and model-based suggestions.
    """
    try:
        result = await mcp_server.analyze_pr(title, description, diff, code)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PR analysis failed: {e}")
