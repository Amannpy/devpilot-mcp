"""
Model Manager — handles lifecycle, initialization, and inference for all supported AI models.

This class provides a unified interface to interact with different LLMs
(e.g., Qwen, OpenAI, local HuggingFace models).
It manages loading, caching, and execution of inference requests
so that the rest of the system can stay model-agnostic.
"""

import os
import asyncio
from typing import Any, Dict, Optional
from backend.core.utils.logger import get_logger
from backend.core.utils.config import AppConfig
from backend.core.models.qwen_model import QwenModel

logger = get_logger(__name__)


class ModelManager:
    """
    Central model management interface for the MCP backend.
    """

    def __init__(self):
        self.config = AppConfig()
        self.models: Dict[str, Any] = {}
        logger.info("Initializing ModelManager...")

        # Preload default model (Qwen or other)
        default_model = self.config.default_model
        self.load_model(default_model)

    # ----------------------------------------------------
    # Model Lifecycle
    # ----------------------------------------------------
    def load_model(self, model_name: str):
        """
        Loads and initializes a model if not already cached.
        """
        if model_name in self.models:
            logger.debug(f"Model '{model_name}' already loaded; skipping re-init.")
            return self.models[model_name]

        logger.info(f"Loading model: {model_name}")
        if model_name.lower().startswith("qwen"):
            model = QwenModel(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.models[model_name] = model
        return model

    def get_model(self, model_name: Optional[str] = None):
        """
        Retrieve a loaded model instance.
        """
        model_name = model_name or self.config.default_model
        if model_name not in self.models:
            self.load_model(model_name)
        return self.models[model_name]

    # ----------------------------------------------------
    # Inference Operations
    # ----------------------------------------------------
    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        Run inference on the selected model.
        """
        model = self.get_model(model_name)
        logger.debug(f"Generating response with model {model_name or self.config.default_model}...")

        try:
            result = await asyncio.to_thread(
                model.generate,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.debug("Model response generated successfully.")
            return result
        except Exception as e:
            logger.exception(f"Model generation failed: {e}")
            return "⚠️ Model failed to generate output. Check logs for details."

    # ----------------------------------------------------
    # Model Utilities
    # ----------------------------------------------------
    def list_loaded_models(self) -> Dict[str, str]:
        """
        List all currently loaded models.
        """
        return {name: type(model).__name__ for name, model in self.models.items()}
