"""
RAG service module for MCP backend.
Handles retrieval-augmented generation (RAG) workflows, optionally used in PR/code analysis.
"""

from typing import Optional, List, Dict
from backend.core.rag.manager import RAGManager
from backend.services.prompt_service import build_prompt
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class RAGService:
    """
    Service layer to handle retrieval-augmented generation (RAG) tasks.
    Can fetch context from indexed documents and provide augmented prompts for the model.
    """

    def __init__(self, enable_rag: bool = True):
        """
        Initialize RAGService.

        Args:
            enable_rag (bool): Whether to use RAG functionality. Defaults to True.
        """
        self.enable_rag = enable_rag
        self.rag_manager = RAGManager() if enable_rag else None

    async def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant context from the RAG index.

        Args:
            query (str): User query or PR/code prompt.
            top_k (int): Number of top results to fetch.

        Returns:
            List[Dict]: List of context documents with metadata.
        """
        if not self.enable_rag or not self.rag_manager:
            logger.warning("RAG is disabled; returning empty context.")
            return []

        logger.info(f"🔍 Retrieving top {top_k} contexts for query: {query[:50]}...")
        results = await self.rag_manager.retrieve(query, top_k=top_k)
        return results

    async def build_augmented_prompt(
        self, user_prompt: str, code: Optional[str] = None, top_k: int = 5
    ) -> str:
        """
        Build a prompt augmented with RAG context.

        Args:
            user_prompt (str): User's natural language query.
            code (Optional[str]): Optional code snippet.
            top_k (int): Number of RAG documents to include.

        Returns:
            str: Structured prompt ready for the model.
        """
        context_docs = await self.retrieve_context(user_prompt, top_k=top_k)
        context_text = "\n\n".join([doc.get("content", "") for doc in context_docs]) if context_docs else None

        prompt = build_prompt(user_prompt, code=code, context=context_text)
        logger.debug(f"Built RAG-augmented prompt (truncated to 200 chars): {prompt[:200]}...")
        return prompt

    async def answer_query(
        self, user_prompt: str, code: Optional[str] = None, top_k: int = 5
    ) -> Dict:
        """
        Full RAG workflow: retrieve context, build prompt, and generate response.

        Returns:
            Dict: Contains structured prompt, retrieved context, and placeholder for model response.
        """
        prompt = await self.build_augmented_prompt(user_prompt, code, top_k=top_k)
        context_docs = await self.retrieve_context(user_prompt, top_k=top_k)

        # Placeholder for future model integration
        model_response = "Response generation to be handled by model."

        return {
            "prompt": prompt,
            "context": context_docs,
            "model_response": model_response,
        }
