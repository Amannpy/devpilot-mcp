# backend/core/mcp_server.py
from backend.services.pr_service import PRService
from backend.services.rag_service import RAGService
from backend.services.session_service import SessionService
from backend.services.prompt_service import build_prompt, detect_intent

class MCPServer:
    """
    Central MCP Server class encapsulating all AI services.
    """

    def __init__(self):
        self.pr_service = PRService()
        self.rag_service = RAGService()
        self.session_service = SessionService()
        self.prompt_service = {"build_prompt": build_prompt, "detect_intent": detect_intent}

    # -------------------------
    # PR Service API
    # -------------------------
    async def analyze_pr(self, title: str, description: str, diff: str = None, code: str = None):
        return await self.pr_service.analyze_pr(title, description, diff, code)

    # -------------------------
    # RAG Service API
    # -------------------------
    async def query_rag(self, query: str, session_id: str = None):
        return await self.rag_service.query(query, session_id=session_id)

    # -------------------------
    # Session Service API
    # -------------------------
    async def start_session(self, user_id: str):
        return await self.session_service.create_session(user_id)

    async def add_message_to_session(self, session_id: str, message: str, role: str = "user"):
        return await self.session_service.add_message(session_id, message, role)

    async def get_session_history(self, session_id: str):
        return await self.session_service.get_history(session_id)

    # -------------------------
    # Prompt Service API
    # -------------------------
    def build_prompt(self, user_prompt: str, code: str = None, context: str = None):
        return self.prompt_service["build_prompt"](user_prompt, code, context)

    def detect_intent(self, user_prompt: str):
        return self.prompt_service["detect_intent"](user_prompt)
