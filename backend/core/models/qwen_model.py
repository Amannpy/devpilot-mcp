"""
QwenModelWrapper — unified async interface for Qwen-based models.
Supports embeddings and text generation for MCP backend.
"""

import asyncio
import hashlib
from typing import List, Optional
import numpy as np

from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class QwenModelWrapper:
    """
    Wrapper for Qwen model (local or remote).
    Handles embeddings and text generation tasks.
    """

    def __init__(self, model_name: str = "Qwen-2.5-7B-Instruct"):
        self.model_name = model_name
        self.is_local = True  # Future: auto-detect remote model availability
        self.embedding_dim = 256  # Set default embedding dimension
        logger.info(
            f"✅ Initialized QwenModelWrapper for {self.model_name} "
            f"with embedding_dim={self.embedding_dim}"
        )

    # ----------------------------------------------------
    # Embeddings
    # ----------------------------------------------------
    async def get_embeddings(self, text: str) -> Optional[List[float]]:
        """
        Compute deterministic embeddings for a given text (mocked for MVP).
        """
        try:
            logger.debug(f"🔹 Generating embeddings for text: {text[:60]!r}...")
            return await asyncio.to_thread(self._compute_embeddings, text)
        except Exception as e:
            logger.exception(f"❌ Embedding generation failed: {e}")
            return None

    def _compute_embeddings(self, text: str) -> List[float]:
        """
        CPU-bound embedding generation.
        Pads or truncates to self.embedding_dim
        """
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.frombuffer(hash_bytes, dtype=np.uint8).astype(float)
        normalized = np.zeros(self.embedding_dim, dtype=float)
        normalized[:min(len(vector), self.embedding_dim)] = vector[:self.embedding_dim] / 255.0
        return normalized.tolist()

    # ----------------------------------------------------
    # Text Generation
    # ----------------------------------------------------
    async def generate_text(
        self,
        prompt: str,
        task_type: str = "general",
        embeddings: Optional[List[float]] = None,
        max_tokens: int = 300,
    ) -> str:
        """
        Asynchronous text generation entrypoint.
        Simulates LLM responses.
        """
        try:
            logger.info(f"🤖 Generating text [task={task_type}, tokens={max_tokens}]")
            return await asyncio.to_thread(
                self._simulate_generation, prompt, task_type, embeddings, max_tokens
            )
        except Exception as e:
            logger.exception(f"❌ Text generation failed: {e}")
            return "⚠️ Text generation failed due to an internal error."

    def _simulate_generation(
        self,
        prompt: str,
        task_type: str,
        embeddings: Optional[List[float]],
        max_tokens: int,
    ) -> str:
        """
        Lightweight simulated model response (replace with actual inference engine later).
        """
        prompt_lower = prompt.lower()

        if "optimize" in prompt_lower or "refactor" in prompt_lower:
            return (
                "✅ Suggested optimization:\n"
                "Use vectorized NumPy operations instead of loops where possible. "
                "Consider memoization or caching of repeated computations."
            )

        elif "bug" in prompt_lower or "error" in prompt_lower:
            return (
                "⚠️ Potential bug detected:\n"
                "Review exception handling and variable scope. "
                "Ensure all input parameters are validated before use."
            )

        elif "document" in prompt_lower or "explain" in prompt_lower:
            return (
                "🧠 Code explanation:\n"
                "This component manages AI-assisted reasoning for pull requests, "
                "documenting key logic paths and ensuring clarity in implementation."
            )

        else:
            return (
                f"🤖 General response from {self.model_name}:\n"
                f"Processed request successfully with task type '{task_type}'."
            )
