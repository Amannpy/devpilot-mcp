"""
Session-related API routes for MCP backend.
"""

from fastapi import APIRouter, Depends
from backend.services.session_service import SessionService

router = APIRouter(prefix="/session", tags=["session"])

# Singleton session service
session_service = SessionService()


@router.post("/start")
async def start_session(user_id: str):
    """
    Start a new user session.
    """
    session_id = await session_service.create_session(user_id)
    return {"session_id": session_id, "message": "Session started successfully"}


@router.post("/history")
async def add_to_history(session_id: str, message: str, role: str = "user"):
    """
    Add a message to session history.
    """
    await session_service.add_message(session_id, message, role)
    return {"message": "Message added to session history"}


@router.get("/history")
async def get_history(session_id: str):
    """
    Retrieve full session history.
    """
    history = await session_service.get_history(session_id)
    return {"session_id": session_id, "history": history}
