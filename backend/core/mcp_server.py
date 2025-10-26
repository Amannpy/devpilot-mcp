"""
MCPServer: The central orchestration layer for all AI and RAG services.
Bridges PR analysis, retrieval-augmented generation, chat sessions, and prompt engineering.
"""

from fastapi import Depends
from backend.services.pr_service import PRService
from backend.services.rag_service import RAGService
from backend.services.session_service import SessionService
from backend.services.prompt_service import build_prompt, detect_intent
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class MCPServer:
    """
    The central orchestrator coordinating PR analysis, RAG, session handling, and prompt intelligence.
    """

    def __init__(self):
        logger.info("🚀 Initializing MCPServer services...")
        self.pr_service = PRService()
        self.rag_service = RAGService()
        self.session_service = SessionService()
        self.prompt_service = {
            "build_prompt": build_prompt,
            "detect_intent": detect_intent
        }
        logger.info("✅ MCPServer initialized successfully")

    # ------------------------------------------------------------------
    # 🧩 PR Service API
    # ------------------------------------------------------------------
    async def analyze_pr(self, title: str, description: str, diff: str | None = None, code: str | None = None):
        logger.debug(f"Analyzing PR: {title}")
        return await self.pr_service.analyze_pr(title, description, diff, code)

    # ------------------------------------------------------------------
    # 📚 RAG Service API
    # ------------------------------------------------------------------
    async def ingest_document(self, content: str, file_type: str = "text"):
        """
        Ingests raw content into the RAG knowledge base.
        Supports text, pdf, and code document types.
        """
        logger.info(f"Ingesting document into RAG: type={file_type}, length={len(content)} chars")
        return await self.rag_service.ingest(content, file_type=file_type)

    async def query_rag(self, query: str, session_id: str | None = None):
        """
        Queries the RAG pipeline for contextual answers or knowledge retrieval.
        """
        logger.debug(f"Querying RAG with query='{query[:50]}...' session_id={session_id}")
        return await self.rag_service.query(query, session_id=session_id)

    # ------------------------------------------------------------------
    # 💬 Session Service API
    # ------------------------------------------------------------------
    async def start_session(self, user_id: str):
        """
        Starts a new chat session and returns a unique session ID.
        """
        logger.info(f"Starting new session for user_id={user_id}")
        return await self.session_service.create_session(user_id)

    async def add_message_to_session(self, session_id: str, message: str, role: str = "user"):
        """
        Adds a message to an existing chat session.
        """
        logger.debug(f"Adding message to session {session_id} (role={role})")
        return await self.session_service.add_message(session_id, message, role)

    async def get_session_history(self, session_id: str):
        """
        Retrieves message history for the specified session.
        """
        logger.debug(f"Retrieving history for session_id={session_id}")
        return await self.session_service.get_history(session_id)

    # ------------------------------------------------------------------
    # 💡 Prompt Service API
    # ------------------------------------------------------------------
    def build_prompt(self, user_prompt: str, code: str | None = None, context: str | None = None):
        """
        Builds a structured AI prompt for code understanding or generation tasks.
        """
        logger.debug(f"Building prompt for intent detection: '{user_prompt[:40]}...'")
        return self.prompt_service["build_prompt"](user_prompt, code, context)

    def detect_intent(self, user_prompt: str):
        """
        Detects the user's intent (explain, optimize, debug, etc.) from a free-form prompt.
        """
        intent = self.prompt_service["detect_intent"](user_prompt)
        logger.debug(f"Detected user intent: {intent}")
        return intent


# ------------------------------------------------------------------
# 🌐 Dependency injection for FastAPI routes
# ------------------------------------------------------------------
_mcp_server_instance: MCPServer | None = None


def get_mcp_server() -> MCPServer:
    """
    Provides a singleton MCPServer instance for FastAPI's dependency injection.
    Ensures all routes use the same initialized services.
    """
    global _mcp_server_instance
    if _mcp_server_instance is None:
        _mcp_server_instance = MCPServer()
    return _mcp_server_instance


# FastAPI dependency alias
MCPServerDep = Depends(get_mcp_server)
