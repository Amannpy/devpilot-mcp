"""
Model Manager — handles lifecycle, initialization, and inference for all supported AI models.
"""

import asyncio
from typing import Any, Dict, Optional
from backend.core.utils.logger import get_logger
from backend.core.utils.config import settings
from backend.core.models.qwen_model import QwenModelWrapper

logger = get_logger(__name__)

class ModelManager:
    """
    Central model management interface for the MCP backend.
    """

    def __init__(self):
        self.config = settings
        self.models: Dict[str, Any] = {}
        logger.info("Initializing ModelManager...")

        # Preload default model
        self.load_model(self.config.default_model)

    # ---------------------------
    # Model Lifecycle
    # ---------------------------
    def load_model(self, model_name: str):
        if model_name in self.models:
            logger.debug(f"Model '{model_name}' already loaded.")
            return self.models[model_name]

        logger.info(f"Loading model: {model_name}")
        if model_name.lower().startswith("qwen"):
            model = QwenModelWrapper(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.models[model_name] = model
        return model

    def get_model(self, model_name: Optional[str] = None):
        model_name = model_name or self.config.default_model
        if model_name not in self.models:
            self.load_model(model_name)
        return self.models[model_name]

    # ---------------------------
    # Inference Operations
    # ---------------------------
    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        model = self.get_model(model_name)
        try:
            return await asyncio.to_thread(
                model.generate_text,
                prompt,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.exception(f"Model generation failed: {e}")
            return "⚠️ Model failed to generate output."

    # ---------------------------
    # Utilities
    # ---------------------------
    def list_loaded_models(self) -> Dict[str, str]:
        return {name: type(model).__name__ for name, model in self.models.items()}

# Singleton instance
model_manager = ModelManager()
