"""
Prompt construction utilities for RAG pipeline.

Responsible for dynamically building prompts based on retrieved
context, user intent, and task type (e.g., PR review, Q&A, summarization).
"""

from typing import List, Dict, Optional
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """
    Handles intelligent prompt construction for model queries.
    Combines retrieved knowledge with user queries for optimal LLM responses.
    """

    def __init__(self):
        # Define different prompt styles for different task types
        self.templates = {
            "qa": (
                "You are an expert assistant. Use the context below to answer accurately.\n\n"
                "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
            ),
            "summarize": (
                "Summarize the following information concisely and clearly.\n\n"
                "Context:\n{context}\n\nSummary:"
            ),
            "pr_review": (
                "You are an AI code review assistant. Analyze the pull request context below, "
                "point out potential issues, and suggest improvements.\n\n"
                "PR Context:\n{context}\n\nInstructions:\n{query}\n\nReview Feedback:"
            ),
            "general": (
                "You are a helpful assistant. Use the provided context to generate an informed response.\n\n"
                "Context:\n{context}\n\nUser Query:\n{query}\n\nResponse:"
            ),
        }

    def _truncate_context(self, context: str, max_chars: int = 4000) -> str:
        """Trim context to avoid overly long prompts."""
        if len(context) > max_chars:
            logger.debug(f"Truncating context from {len(context)} to {max_chars} characters.")
            return context[:max_chars] + "..."
        return context

    def build_prompt(
        self,
        query: str,
        context_chunks: Optional[List[Dict]] = None,
        task_type: str = "general",
        include_metadata: bool = False,
    ) -> str:
        """
        Build a formatted prompt for model input.

        Args:
            query (str): User's query or instruction.
            context_chunks (List[Dict]): Retrieved context from FAISS or DB.
            task_type (str): Type of prompt (qa, summarize, pr_review, general).
            include_metadata (bool): Whether to include metadata like filenames/line numbers.

        Returns:
            str: A structured and optimized prompt for the LLM.
        """
        if not context_chunks:
            logger.warning("No context provided for prompt generation.")
            context = "(No additional context available)"
        else:
            formatted_contexts = []
            for chunk in context_chunks:
                context_part = chunk.get("content", "")
                if include_metadata:
                    meta = chunk.get("metadata", {})
                    source = meta.get("source", "unknown")
                    context_part = f"[Source: {source}]\n{context_part}"
                formatted_contexts.append(context_part)

            context = "\n\n---\n\n".join(formatted_contexts)

        context = self._truncate_context(context)

        template = self.templates.get(task_type, self.templates["general"])
        prompt = template.format(context=context.strip(), query=query.strip())

        logger.debug(f"Prompt built for task: {task_type} | Length: {len(prompt)} chars")
        return prompt

    def build_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        context_chunks: Optional[List[Dict]] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Builds a chat-style prompt combining previous messages and optional context.

        Args:
            messages (List[Dict]): [{"role": "user"/"assistant", "content": "..."}]
            context_chunks (List[Dict]): Optional retrieved data.
            system_instruction (str): Optional system role definition.

        Returns:
            str: Chat-style conversation formatted for LLM.
        """
        conversation = []

        if system_instruction:
            conversation.append(f"System: {system_instruction}")

        for msg in messages:
            conversation.append(f"{msg['role'].capitalize()}: {msg['content']}")

        conversation_text = "\n".join(conversation)

        if context_chunks:
            context_text = "\n\n".join([c.get("content", "") for c in context_chunks])
            prompt = f"{conversation_text}\n\n---\n\nRelevant Context:\n{context_text}\n\nContinue the conversation:"
        else:
            prompt = f"{conversation_text}\n\nContinue the conversation:"

        logger.debug("Chat prompt constructed successfully.")
        return prompt
