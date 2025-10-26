"""
End-to-End Inference Test for MCP AI Backend
Tests ModelManager, Qwen model, embeddings, and RAG functionality.
"""

import asyncio
from backend.core.models.model_manager import ModelManager
from backend.config import USE_RAG


async def test_model_generation():
    print("🧪 Starting inference test...\n")

    # Initialize model manager
    model_manager = ModelManager()

    # ----------------------
    # Test text generation
    # ----------------------
    prompt = "Explain how to optimize a Python loop."
    print(f"Prompt: {prompt}\n")

    result = await model_manager.generate(prompt=prompt)
    print(f"Generated Text:\n{result}\n")

    # ----------------------
    # Test embeddings
    # ----------------------
    model = model_manager.get_model()
    embeddings = await model.get_embeddings(prompt)
    print(f"Embeddings (first 10 values): {embeddings[:10]}\n")
    print(f"Embedding length: {len(embeddings)}\n")

    # ----------------------
    # Optional RAG test
    # ----------------------
    if USE_RAG:
        try:
            from backend.core.rag.manager import RAGManager
            rag_manager = RAGManager()
            print("RAGManager initialized successfully.")
        except Exception as e:
            print(f"⚠️ RAGManager initialization failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_model_generation())
