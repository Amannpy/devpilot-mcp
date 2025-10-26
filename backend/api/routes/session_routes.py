"""
Session management API routes for the MCP backend.
Handles creation, message history tracking, and retrieval of chat sessions.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from backend.services.session_service import SessionService
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/session", tags=["Session"])

session_service = SessionService()


# ----------------------------
# Pydantic Schemas
# ----------------------------
class StartSessionRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user starting the session")


class AddMessageRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to which the message belongs")
    message: str = Field(..., description="Message text to append to history")
    role: str = Field("user", description="Role of the message sender (e.g., user, system, assistant)")


# ----------------------------
# Routes
# ----------------------------
@router.post("/start")
async def start_session(request: StartSessionRequest):
    """
    Start a new chat session for a user.
    """
    try:
        logger.info(f"🟢 Starting new session for user_id={request.user_id}")
        session_id = await session_service.create_session(request.user_id)
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Session started successfully",
        }
    except Exception as e:
        logger.exception(f"❌ Failed to start session for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start session")


@router.post("/history")
async def add_to_history(request: AddMessageRequest):
    """
    Add a message to an existing session’s history.
    """
    try:
        logger.info(f"🗨️ Adding message to session {request.session_id} by role={request.role}")
        await session_service.add_message(request.session_id, request.message, request.role)
        return {"status": "success", "message": "Message added to session history"}
    except Exception as e:
        logger.exception(f"❌ Failed to add message to session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to add message to history")


@router.get("/history")
async def get_history(session_id: str):
    """
    Retrieve the full chat history for a given session.
    """
    try:
        logger.info(f"📜 Fetching chat history for session_id={session_id}")
        history = await session_service.get_history(session_id)
        if history is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "status": "success",
            "session_id": session_id,
            "history": history,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Failed to fetch session history for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session history")
