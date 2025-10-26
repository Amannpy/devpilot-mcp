"""
Service layer for managing chat sessions and history.
Handles session creation, message logging, and retrieval.
"""

import uuid
from typing import Dict, List, Any
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)

# In-memory session storage for MVP
_sessions: Dict[str, List[Dict[str, Any]]] = {}


class SessionService:
    """Service managing user chat sessions and history."""

    async def create_session(self, user_id: str) -> str:
        """
        Create a new session for a given user.

        Args:
            user_id (str): Unique identifier of the user.

        Returns:
            str: Session ID.
        """
        session_id = str(uuid.uuid4())
        _sessions[session_id] = []
        logger.info(f"🟢 Created new session {session_id} for user {user_id}")
        return session_id

    async def add_message(self, session_id: str, message: str, role: str = "user"):
        """
        Add a message to a session's history.

        Args:
            session_id (str): Session ID.
            message (str): Message content.
            role (str): Role of sender (user/system/assistant). Defaults to 'user'.

        Raises:
            ValueError: If the session ID does not exist.
        """
        if session_id not in _sessions:
            logger.warning(f"❌ Attempt to add message to non-existent session {session_id}")
            raise ValueError(f"Session ID {session_id} not found")

        _sessions[session_id].append({"role": role, "message": message})
        logger.debug(f"🗨️ Added message to session {session_id} by role={role}")

    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve the full message history for a session.

        Args:
            session_id (str): Session ID.

        Returns:
            List[Dict[str, Any]]: Ordered list of messages (role + message).

        Raises:
            ValueError: If the session ID does not exist.
        """
        if session_id not in _sessions:
            logger.warning(f"❌ Attempt to fetch history for non-existent session {session_id}")
            raise ValueError(f"Session ID {session_id} not found")

        logger.info(f"📜 Retrieved history for session {session_id}")
        return _sessions[session_id]
