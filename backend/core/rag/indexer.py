"""
RAG Indexer — builds and manages FAISS vector stores from ingested content.
"""

import os
import faiss
import json
import numpy as np
from typing import List, Dict, Optional
from backend.core.utils.logger import get_logger
from backend.core.models.model_manager import model_manager
from backend.core.rag.ingest.code import CodePreprocessor
from backend.core.rag.ingest.pdf import PDFPreprocessor
from backend.core.rag.ingest.text import TextPreprocessor

logger = get_logger(__name__)


class RAGIndexer:
    """Build and manage FAISS indexes for RAG retrieval."""

    def __init__(self, index_dir: str = "faiss_index"):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "rag_index.faiss")
        self.metadata_path = os.path.join(index_dir, "metadata.json")
        self._load_existing_index()

    def _load_existing_index(self):
        """Load existing FAISS index or initialize new."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info("Loaded existing FAISS index.")
            else:
                self.index = faiss.IndexFlatL2(model_manager.embedding_dim)
                logger.info("Initialized new FAISS index.")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self.index = faiss.IndexFlatL2(model_manager.embedding_dim)

        self.metadata = self._load_metadata()

    def _load_metadata(self) -> List[Dict]:
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        return []

    async def _generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for content chunks."""
        texts = [chunk["content"] for chunk in chunks]
        embeddings = await model_manager.get_batch_embeddings(texts)
        return np.array(embeddings).astype("float32")

    async def index_files(self, file_paths: List[str]):
        """Index multiple files."""
        code_ingestor = CodePreprocessor()
        pdf_ingestor = PDFPreprocessor()
        text_ingestor = TextPreprocessor()

        new_chunks = []

        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue

            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                chunks = pdf_ingestor.extract_and_clean(path)
            elif ext in code_ingestor.supported_extensions:
                chunks = code_ingestor.extract_code_chunks(path)
            else:
                chunks = text_ingestor.extract_text_chunks(path)

            new_chunks.extend(chunks)

        if not new_chunks:
            logger.warning("No chunks to index.")
            return {"indexed": 0, "message": "No content extracted"}

        embeddings = await self._generate_embeddings(new_chunks)
        self.index.add(embeddings)
        self.metadata.extend(new_chunks)

        self._persist_index()
        self._persist_metadata()

        logger.info(f"Indexed {len(new_chunks)} chunks.")
        return {"indexed": len(new_chunks), "message": "Files indexed successfully"}

    def _persist_index(self):
        """Save FAISS index."""
        try:
            faiss.write_index(self.index, self.index_path)
            logger.debug("FAISS index saved.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def _persist_metadata(self):
        """Save metadata alongside FAISS index."""
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.debug("Metadata saved.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search FAISS index and return top-k relevant chunks."""
        query_emb = np.array(await model_manager.get_code_embeddings(query)).reshape(1, -1).astype("float32")
        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                meta["score"] = float(distances[0][i])
                results.append(meta)

        return sorted(results, key=lambda x: x["score"])
