"""
Retriever module for RAG pipeline.
Handles context retrieval from FAISS vector store and fallback text matching.
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from backend.core.utils.logger import get_logger
from backend.core.models.qwen_model import QwenModelWrapper

logger = get_logger(__name__)


class RAGRetriever:
    """
    Retrieves relevant context chunks for a given query using FAISS or simple heuristics.
    """

    def __init__(
        self,
        faiss_index_dir: str = "faiss_index",
        top_k: int = 5,
    ):
        self.faiss_index_dir = faiss_index_dir
        self.top_k = top_k
        self.model = QwenModelWrapper()
        self.index = None
        self.documents: List[Dict] = []
        self._load_index()

    # ---------------------------------------------------------------------
    # Initialization and Index Loading
    # ---------------------------------------------------------------------
    def _load_index(self):
        """
        Load FAISS index and associated metadata if available.
        """
        try:
            import faiss
            index_path = os.path.join(self.faiss_index_dir, "rag_index.faiss")
            meta_path = os.path.join(self.faiss_index_dir, "rag_metadata.npy")

            if os.path.exists(index_path) and os.path.exists(meta_path):
                logger.info(f"Loading FAISS index from {index_path}")
                self.index = faiss.read_index(index_path)
                self.documents = np.load(meta_path, allow_pickle=True).tolist()
            else:
                logger.warning("FAISS index or metadata not found; using fallback retriever.")
        except ImportError:
            logger.warning("FAISS not installed; retrieval will use fallback search.")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")

    # ---------------------------------------------------------------------
    # Core Retrieval Logic
    # ---------------------------------------------------------------------
    async def retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve the most relevant context for a given query.
        """
        top_k = top_k or self.top_k
        results: List[Dict] = []

        if self.index is not None and self.documents:
            try:
                query_embedding = await self.model.get_embeddings(query)
                if query_embedding is None:
                    logger.warning("Model returned None for embeddings; fallback to keyword search.")
                    return self._fallback_retrieval(query)

                # Convert to numpy and perform FAISS search
                import faiss
                query_vector = np.array([query_embedding]).astype("float32")
                distances, indices = self.index.search(query_vector, top_k)

                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.documents):
                        continue
                    doc = self.documents[idx]
                    doc["score"] = float(dist)
                    results.append(doc)

                logger.info(f"Retrieved {len(results)} chunks from FAISS index.")
                return results
            except Exception as e:
                logger.error(f"Retrieval error using FAISS: {e}")
                return self._fallback_retrieval(query)
        else:
            logger.warning("FAISS index unavailable; using fallback retrieval.")
            return self._fallback_retrieval(query)

    # ---------------------------------------------------------------------
    # Fallback Search
    # ---------------------------------------------------------------------
    def _fallback_retrieval(self, query: str) -> List[Dict]:
        """
        Fallback retrieval via keyword matching over local text files.
        """
        fallback_dir = "data"
        context_results: List[Dict] = []

        if not os.path.exists(fallback_dir):
            logger.warning("No fallback directory found for text search.")
            return [{"content": "No context found.", "metadata": {}}]

        for root, _, files in os.walk(fallback_dir):
            for file in files:
                if not file.endswith((".py", ".txt", ".md")):
                    continue
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    if query.lower() in content.lower():
                        snippet = self._extract_relevant_snippet(content, query)
                        context_results.append(
                            {"content": snippet, "metadata": {"source": file_path}}
                        )
                except Exception as e:
                    logger.error(f"Error reading file {file}: {e}")

        if not context_results:
            context_results.append({"content": "No relevant context found.", "metadata": {}})

        logger.info(f"Fallback retrieved {len(context_results)} context chunks.")
        return context_results

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------
    def _extract_relevant_snippet(self, content: str, query: str, window: int = 200) -> str:
        """
        Extract a snippet around the first occurrence of the query term.
        """
        idx = content.lower().find(query.lower())
        if idx == -1:
            return content[:window]
        start = max(0, idx - window)
        end = min(len(content), idx + window)
        return content[start:end]
