"""
Model Manager — handles lifecycle, initialization, and inference for all supported AI models.
"""
import asyncio
from typing import Any, Dict, Optional
from backend.core.utils.logger import get_logger
from backend.config import settings  # ✅ Fixed import
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

        # Preload default model (optional - can be lazy loaded)
        try:
            self.load_model(self.config.default_model)
            logger.info(f"✅ Default model '{self.config.default_model}' preloaded")
        except Exception as e:
            logger.warning(f"Could not preload default model: {e}")
            logger.info("Models will be loaded on first use")

    # ---------------------------
    # Model Lifecycle
    # ---------------------------
    def load_model(self, model_name: str):
        """
        Load a model by name

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model instance
        """
        if model_name in self.models:
            logger.debug(f"Model '{model_name}' already loaded.")
            return self.models[model_name]

        logger.info(f"Loading model: {model_name}")

        # Determine model type and load
        if "qwen" in model_name.lower():
            model = QwenModelWrapper(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.models[model_name] = model
        logger.info(f"✅ Model '{model_name}' loaded successfully")
        return model

    def get_model(self, model_name: Optional[str] = None):
        """
        Get a loaded model or load it if not already loaded

        Args:
            model_name: Name of the model (defaults to default_model)

        Returns:
            Model instance
        """
        model_name = model_name or self.config.default_model

        if model_name not in self.models:
            self.load_model(model_name)

        return self.models[model_name]

    def unload_model(self, model_name: str):
        """
        Unload a model to free memory

        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Model '{model_name}' unloaded")

            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared")
            except ImportError:
                pass

    # ---------------------------
    # Inference Operations
    # ---------------------------
    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate text asynchronously

        Args:
            prompt: Input prompt
            model_name: Model to use (defaults to default_model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        model = self.get_model(model_name)

        try:
            # Run in thread pool to avoid blocking
            return await asyncio.to_thread(
                model.generate_text,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.exception(f"Model generation failed: {e}")
            return f"⚠️ Model failed to generate output: {str(e)}"

    def generate_sync(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Generate text synchronously

        Args:
            prompt: Input prompt
            model_name: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        model = self.get_model(model_name)

        try:
            return model.generate_text(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.exception(f"Model generation failed: {e}")
            return f"⚠️ Model failed to generate output: {str(e)}"

    async def chat(
        self,
        messages: list[Dict[str, str]],
        model_name: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Chat interface with message history

        Args:
            messages: List of message dicts with 'role' and 'content'
            model_name: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Assistant response
        """
        model = self.get_model(model_name)

        try:
            return await asyncio.to_thread(
                model.chat,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.exception(f"Chat failed: {e}")
            return f"⚠️ Chat failed: {str(e)}"

    # ---------------------------
    # Utilities
    # ---------------------------
    def list_loaded_models(self) -> Dict[str, str]:
        """
        Get list of currently loaded models

        Returns:
            Dict mapping model names to their class names
        """
        return {name: type(model).__name__ for name, model in self.models.items()}

    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded

        Args:
            model_name: Name of the model

        Returns:
            True if loaded, False otherwise
        """
        return model_name in self.models

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model

        Args:
            model_name: Name of the model

        Returns:
            Dict with model information
        """
        model_name = model_name or self.config.default_model

        info = {
            "name": model_name,
            "loaded": self.is_model_loaded(model_name),
            "device": self.config.get_device(),
        }

        if self.is_model_loaded(model_name):
            model = self.models[model_name]
            info["type"] = type(model).__name__

        return info

    def __repr__(self) -> str:
        return f"ModelManager(loaded_models={list(self.models.keys())})"


# Singleton instance
model_manager = ModelManager()