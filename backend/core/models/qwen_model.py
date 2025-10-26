"""
QwenModel — unified wrapper around local or remote Qwen models.
Supports embeddings and text generation for MCP backend.
"""

import asyncio
from typing import List, Optional, Dict, Any
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class QwenModel:
    """
    Wrapper for Qwen model (local or remote).
    Handles text generation and embeddings.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Initializing Qwen model: {model_name}")

        # Placeholder: detect backend (local vs API)
        # In production, this can dynamically load from HuggingFace or local engine
        self.is_local = True  # Switch if using remote API

    # ----------------------------------------------------
    # Embeddings
    # ----------------------------------------------------
    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """
        Compute embeddings for given text.
        """
        try:
            # Simulate embedding computation
            import numpy as np
            import hashlib

            # Simple deterministic embedding generator for MVP
            hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.frombuffer(hash_bytes, dtype=np.uint8).astype(float)
            normalized = vector[:256] / 255.0
            return normalized.tolist()

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    # ----------------------------------------------------
    # Text Generation
    # ----------------------------------------------------
    def generate(
        self,
        prompt: str,
        task_type: str = "general",
        embeddings: Optional[List[float]] = None,
        temperature: float = 0.7,
        max_tokens: int = 300,
    ) -> str:
        """
        Generate text from the Qwen model.
        """
        try:
            # For MVP, simulate generation; later connect with real inference
            logger.info(f"Generating text with {self.model_name} [task={task_type}]")

            if "optimize" in prompt.lower():
                return (
                    "✅ Suggested optimization:\n"
                    "Use vectorized NumPy operations instead of loops where possible. "
                    "Consider caching intermediate computations."
                )
            elif "bug" in prompt.lower():
                return (
                    "⚠️ Potential bug detected:\n"
                    "Check variable scoping and input validation. Missing try/except may cause runtime errors."
                )
            elif "document" in prompt.lower() or "explain" in prompt.lower():
                return (
                    "🧠 Code explanation:\n"
                    "This module trains a simple regression model on housing data. "
                    "It loads the dataset, splits it, trains a LinearRegression model, "
                    "and evaluates it using Mean Squared Error."
                )
            else:
                return (
                    f"🤖 General response from {self.model_name}:\n"
                    f"'{prompt[:200]}...' — processed successfully."
                )

        except Exception as e:
            logger.exception(f"Qwen text generation failed: {e}")
            return "⚠️ Text generation failed due to internal error."

    # ----------------------------------------------------
    # Async wrapper
    # ----------------------------------------------------
    async def generate_text(
        self,
        prompt: str,
        task_type: str = "general",
        embeddings: Optional[List[float]] = None,
        max_tokens: int = 300,
    ) -> str:
        """
        Async version for integration with FastAPI or async services.
        """
        return await asyncio.to_thread(
            self.generate,
            prompt=prompt,
            task_type=task_type,
            embeddings=embeddings,
            max_tokens=max_tokens,
        )
