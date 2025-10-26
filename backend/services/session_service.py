"""
Service layer for managing chat sessions and history.
"""

import uuid
from typing import Dict, List, Any

# In-memory session storage for MVP
_sessions: Dict[str, List[Dict[str, Any]]] = {}


class SessionService:
    async def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = []
        return session_id

    async def add_message(self, session_id: str, message: str, role: str = "user"):
        if session_id not in _sessions:
            raise ValueError("Session ID not found")
        _sessions[session_id].append({"role": role, "message": message})

    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        return _sessions.get(session_id, [])
