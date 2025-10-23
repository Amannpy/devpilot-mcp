"""
FAISS indexer for vector storage and retrieval.
"""

import os
import pickle
import json
import faiss
import numpy as np
from typing import List, Dict, Optional

from src.models import model_manager


class Indexer:
    """Manages FAISS index and metadata storage."""

    def __init__(
            self,
            embedding_dim: Optional[int] = None,
            persist_dir: Optional[str] = None
    ):
        self.persist_dir = persist_dir or "./faiss_index"
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index_file_path = os.path.join(self.persist_dir, "index.faiss")
        self.meta_file_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.config_file_path = os.path.join(self.persist_dir, "config.json")

        # Load or set embedding dimension
        self.embedding_dim = embedding_dim or self._load_embedding_dim()

        if self.embedding_dim <= 0:
            self.embedding_dim = 768  # Default fallback

        # Initialize FAISS index
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata: List[Dict] = []

        # Try to load existing index
        self.load()

    def _load_embedding_dim(self) -> int:
        """Load embedding dimension from config."""
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, "r") as f:
                    cfg = json.load(f)
                    return cfg.get("embedding_dim", 0)
            except Exception:
                pass
        return 0

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for list of texts.
        Returns numpy array of shape (len(texts), embedding_dim).
        """
        embeddings = []

        for text in texts:
            try:
                emb = await model_manager.get_code_embeddings(text)

                if emb is None or len(emb) == 0:
                    # Fallback: zero vector
                    emb = np.zeros(self.embedding_dim, dtype=np.float32)
                else:
                    emb = np.array(emb, dtype=np.float32)

                    # Ensure correct dimension
                    if len(emb) != self.embedding_dim:
                        # Pad or truncate
                        if len(emb) < self.embedding_dim:
                            emb = np.pad(emb, (0, self.embedding_dim - len(emb)))
                        else:
                            emb = emb[:self.embedding_dim]

                embeddings.append(emb)

            except Exception as e:
                # On error, use zero vector
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

        return np.stack(embeddings, axis=0)

    async def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to the index.
        Each chunk should have at least 'text' field.
        """
        if not chunks:
            return

        # Extract texts
        texts = [c.get("text", "") for c in chunks]

        # Generate embeddings
        embeddings = await self.embed_texts(texts)

        # Add to FAISS
        self.faiss_index.add(embeddings)

        # Store metadata
        self.metadata.extend(chunks)

    def persist(self):
        """Save index and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.faiss_index, self.index_file_path)

        # Save metadata
        with open(self.meta_file_path, "wb") as f:
            pickle.dump(self.metadata, f)

        # Save config
        with open(self.config_file_path, "w") as f:
            json.dump({"embedding_dim": self.embedding_dim}, f)

    def load(self):
        """Load index and metadata from disk."""
        # Load metadata
        if os.path.exists(self.meta_file_path):
            try:
                with open(self.meta_file_path, "rb") as f:
                    self.metadata = pickle.load(f)
            except Exception:
                self.metadata = []

        # Load config
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, "r") as f:
                    cfg = json.load(f)
                    self.embedding_dim = cfg.get("embedding_dim", self.embedding_dim)
            except Exception:
                pass

        # Load FAISS index
        if os.path.exists(self.index_file_path):
            try:
                self.faiss_index = faiss.read_index(self.index_file_path)
            except Exception:
                # Create new index if load fails
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

    def clear(self):
        """Clear the index an