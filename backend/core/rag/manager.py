"""
RAG Manager — orchestrates retrieval-augmented generation across indexing and model layers.
"""

import asyncio
from typing import List, Dict, Optional
from backend.core.models.qwen_model import QwenModelWrapper
from backend.core.rag.indexer import RAGIndexer
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class RAGManager:
    """High-level orchestrator for RAG pipelines."""

    def __init__(self):
        self.indexer = RAGIndexer()
        self.model = QwenModelWrapper()

    async def ingest_and_index(self, file_paths: List[str]) -> Dict:
        """Ingest, process, and index new files into the FAISS store."""
        try:
            result = await self.indexer.index_files(file_paths)
            return result
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            return {"error": str(e)}

    async def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k chunks relevant to a query."""
        try:
            results = await self.indexer.search(query, top_k=top_k)
            logger.info(f"Retrieved {len(results)} context chunks for query.")
            return results
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    async def generate_response(self, query: str, context_chunks: Optional[List[Dict]] = None) -> str:
        """Generate an answer based on query and retrieved context."""
        try:
            if not context_chunks:
                context_chunks = await self.retrieve_context(query)

            context_text = "\n\n".join([c["content"] for c in context_chunks[:5]])
            prompt = (
                f"Answer the following question using the provided context.\n\n"
                f"Context:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"
            )

            response = await self.model.generate_text(prompt)
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"Error: {str(e)}"

    async def query(self, query: str, top_k: int = 5) -> Dict:
        """Unified API — retrieve context and generate final answer."""
        retrieved = await self.retrieve_context(query, top_k=top_k)
        answer = await self.generate_response(query, retrieved)
        return {"query": query, "context": retrieved, "answer": answer}
