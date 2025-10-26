"""
Session manager for RAG conversational memory.

Responsibilities:
- create and store ephemeral/persisted chat sessions
- append messages (user/system/assistant) to session history
- retrieve, clear, and list sessions
- persist sessions to disk under rag_db/ for simple durability
- concurrency-safe (per-session asyncio.Lock)
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from backend.core.utils.logger import get_logger

logger = get_logger(__name__)

RAG_DB_DIR = Path("rag_db")
RAG_DB_FILE = RAG_DB_DIR / "sessions.json"
DEFAULT_MAX_HISTORY = 200  # max messages to keep per session


@dataclass
class SessionMessage:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str  # ISO format


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: str
    updated_at: str
    messages: List[SessionMessage]


class SessionService:
    """
    Async session service with simple disk persistence.

    NOTE: This is designed for an MVP. For production at scale use a real datastore
    (Redis, Postgres, or vector DB for semantic memory).
    """

    def __init__(self, persist_file: Optional[Path] = None, max_history: int = DEFAULT_MAX_HISTORY):
        self.persist_file = persist_file or RAG_DB_FILE
        RAG_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history

        # in-memory sessions: session_id -> Session
        self._sessions: Dict[str, Session] = {}

        # per-session locks to avoid concurrent writes
        self._locks: Dict[str, asyncio.Lock] = {}

        # global lock for sessions dict
        self._global_lock = asyncio.Lock()

        # load persisted sessions if available
        self._load_from_disk()

    # -------------------------
    # Persistence helpers
    # -------------------------
    def _load_from_disk(self):
        try:
            if self.persist_file.exists():
                with open(self.persist_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for sid, sdata in data.items():
                    msgs = [SessionMessage(**m) for m in sdata.get("messages", [])]
                    session = Session(
                        session_id=sid,
                        user_id=sdata.get("user_id", "unknown"),
                        created_at=sdata.get("created_at", datetime.utcnow().isoformat()),
                        updated_at=sdata.get("updated_at", datetime.utcnow().isoformat()),
                        messages=msgs,
                    )
                    self._sessions[sid] = session
                    self._locks[sid] = asyncio.Lock()
                logger.info(f"Loaded {len(self._sessions)} sessions from disk.")
        except Exception as e:
            logger.warning(f"Failed to load sessions from disk: {e}")

    def _persist_to_disk(self):
        try:
            serializable = {
                sid: {
                    "user_id": s.user_id,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    "messages": [asdict(m) for m in s.messages],
                }
                for sid, s in self._sessions.items()
            }
            with open(self.persist_file, "w", encoding="utf-8") as fh:
                json.dump(serializable, fh, indent=2, ensure_ascii=False)
            logger.debug("Persisted sessions to disk.")
        except Exception as e:
            logger.error(f"Failed to persist sessions to disk: {e}")

    # -------------------------
    # Utility helpers
    # -------------------------
    def _ensure_lock(self, session_id: str):
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()

    def _trim_history(self, session: Session):
        if len(session.messages) > self.max_history:
            # keep most recent messages
            session.messages = session.messages[-self.max_history :]

    # -------------------------
    # Public API
    # -------------------------
    async def create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """
        Create a new session (or reuse provided session_id).
        Returns the session_id.
        """
        # Generate a session id if not provided
        sid = session_id or f"{user_id}-{int(datetime.utcnow().timestamp() * 1000)}"

        async with self._global_lock:
            if sid in self._sessions:
                logger.info(f"Session {sid} already exists; returning existing session.")
                return sid

            now = datetime.utcnow().isoformat()
            session = Session(session_id=sid, user_id=user_id, created_at=now, updated_at=now, messages=[])
            self._sessions[sid] = session
            self._ensure_lock(sid)
            self._persist_to_disk()
            logger.info(f"Created session {sid} for user {user_id}.")
            return sid

    async def add_message(self, session_id: str, content: str, role: str = "user") -> None:
        """
        Append a message to the session history.
        role should be one of "user", "assistant", "system".
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")

        self._ensure_lock(session_id)
        lock = self._locks[session_id]
        async with lock:
            session = self._sessions[session_id]
            timestamp = datetime.utcnow().isoformat()
            msg = SessionMessage(role=role, content=content, timestamp=timestamp)
            session.messages.append(msg)
            session.updated_at = timestamp
            self._trim_history(session)
            # persist after each append for durability (MVP)
            self._persist_to_disk()
            logger.debug(f"Added message to {session_id} (role={role}).")

    async def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return session history as list of dicts (most recent last).
        If limit provided, returns the last `limit` messages.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")

        # read-only access does not require lock, but to be safe use lock
        self._ensure_lock(session_id)
        async with self._locks[session_id]:
            messages = self._sessions[session_id].messages
            if limit is not None:
                messages = messages[-limit:]
            return [asdict(m) for m in messages]

    async def clear_session(self, session_id: str) -> None:
        """
        Remove messages from a session but keep the session metadata.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")

        async with self._locks.setdefault(session_id, asyncio.Lock()):
            session = self._sessions[session_id]
            session.messages = []
            session.updated_at = datetime.utcnow().isoformat()
            self._persist_to_disk()
            logger.info(f"Cleared session {session_id} history.")

    async def delete_session(self, session_id: str) -> None:
        """
        Delete the entire session from memory and disk.
        """
        async with self._global_lock:
            if session_id in self._sessions:
                self._sessions.pop(session_id, None)
                self._locks.pop(session_id, None)
                # persist deletion
                self._persist_to_disk()
                logger.info(f"Deleted session {session_id}.")
            else:
                raise KeyError(f"Session {session_id} not found")

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Return a list of session metadata (no messages).
        """
        async with self._global_lock:
            out = []
            for sid, s in self._sessions.items():
                out.append(
                    {
                        "session_id": sid,
                        "user_id": s.user_id,
                        "created_at": s.created_at,
                        "updated_at": s.updated_at,
                        "message_count": len(s.messages),
                    }
                )
            return out

    # -------------------------
    # Shutdown hook
    # -------------------------
    async def shutdown(self):
        """
        Persist state and perform cleanup before application exit.
        """
        try:
            self._persist_to_disk()
            logger.info("SessionService shutdown: sessions persisted.")
        except Exception as e:
            logger.error(f"Error persisting sessions on shutdown: {e}")


# Singleton instance for application use
session_service = SessionService()
