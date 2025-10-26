import asyncio
from typing import List, Optional
from backend.core.models.model_manager import model_manager

class QwenModelWrapper:
    """Wrapper for local or HF Qwen model."""

    async def get_embeddings(self, text: str) -> Optional[List[float]]:
        return await model_manager.get_code_embeddings(text)

    async def generate_text(
        self, prompt: str, task_type: str = "general", embeddings: Optional[List[float]] = None, max_tokens: int = 300
    ) -> str:
        return await model_manager.qwen.generate_text(prompt, task_type=task_type, embeddings=embeddings, max_new_tokens=max_tokens)
