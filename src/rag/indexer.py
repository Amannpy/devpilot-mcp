"""
FAISS indexer for vector storage and retrieval.
"""

import os
import pickle
import json
import faiss
import numpy as np
from typing import List, Dict, Optional, Any

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

        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            self.embedding_dim = 768  # Default fallback

        # Initialize FAISS index (L2)
        try:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        except Exception:
            # In case faiss fails to initialize with given dim, fallback to default
            self.embedding_dim = 768
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

        self.metadata: List[Dict[str, Any]] = []

        # Try to load existing index and metadata
        self.load()

    def _load_embedding_dim(self) -> int:
        """Load embedding dimension from config file if present."""
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return int(cfg.get("embedding_dim", 0))
            except Exception:
                # ignore and return 0 (caller will set default)
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
                    arr = np.zeros(self.embedding_dim, dtype=np.float32)
                else:
                    arr = np.array(emb, dtype=np.float32)

                    # Ensure correct dimension
                    if arr.ndim != 1:
                        arr = arr.flatten()

                    if len(arr) != self.embedding_dim:
                        # Pad or truncate appropriately
                        if len(arr) < self.embedding_dim:
                            pad_width = self.embedding_dim - len(arr)
                            arr = np.pad(arr, (0, pad_width), mode="constant", constant_values=0.0)
                        else:
                            arr = arr[: self.embedding_dim]

                embeddings.append(arr)

            except Exception:
                # On any error, append a zero vector for stability
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

        if not embeddings:
            # Return an empty array with correct shape
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        return np.stack(embeddings, axis=0).astype(np.float32)

    async def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks to the index. Each chunk should have a 'text' field and optional metadata.
        This method embeds chunk texts and appends them to the FAISS index and metadata store.
        """
        if not chunks:
            return

        # Extract texts safely
        texts = [c.get("text", "") if isinstance(c, dict) else "" for c in chunks]

        # Generate embeddings
        embeddings = await self.embed_texts(texts)

        # If no embeddings produced, avoid adding
        if embeddings.shape[0] == 0:
            return

        try:
            # If index dimension differs from the embeddings, recreate index
            if self.faiss_index.d != embeddings.shape[1]:
                # Recreate index with the new embedding dim (attempt fallback)
                self.embedding_dim = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

            # Add embeddings into FAISS
            self.faiss_index.add(embeddings)

            # Store metadata aligned with embeddings order
            self.metadata.extend(chunks)

        except Exception as e:
            # If anything goes wrong, do not crash — log and continue
            # In a real app, you might want to surface/log this to a monitoring system
            print(f"⚠️  Indexer.add_chunks error: {e}")

    def persist(self):
        """Save index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, self.index_file_path)
        except Exception as e:
            # writing index might fail (permissions, invalid index) -> warn but continue
            print(f"⚠️  Could not persist FAISS index: {e}")

        try:
            # Save metadata
            with open(self.meta_file_path, "wb") as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"⚠️  Could not persist metadata: {e}")

        try:
            # Save config (embedding dim)
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                json.dump({"embedding_dim": int(self.embedding_dim)}, f)
        except Exception as e:
            print(f"⚠️  Could not persist config: {e}")

    def load(self):
        """Load index and metadata from disk (if present)."""
        # Load metadata
        if os.path.exists(self.meta_file_path):
            try:
                with open(self.meta_file_path, "rb") as f:
                    self.metadata = pickle.load(f) or []
            except Exception:
                self.metadata = []

        # Load config for embedding_dim
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if isinstance(cfg.get("embedding_dim"), int) and cfg["embedding_dim"] > 0:
                        self.embedding_dim = cfg["embedding_dim"]
            except Exception:
                # ignore and keep current embedding_dim
                pass

        # Load FAISS index
        if os.path.exists(self.index_file_path):
            try:
                idx = faiss.read_index(self.index_file_path)
                # If loaded index dimension doesn't match embedding_dim, override embedding_dim
                try:
                    dim = idx.d
                    if dim != self.embedding_dim:
                        self.embedding_dim = dim
                except Exception:
                    pass
                self.faiss_index = idx
            except Exception:
                # Create a new index if load fails
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            # Create a fresh index
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

    def search_by_embeddings(self, query_embeddings: np.ndarray, k: int = 6) -> Dict[str, List]:
        """
        Query FAISS index with precomputed embeddings.
        Returns dictionary: {'distances': [...], 'indices': [...]}
        """
        if query_embeddings is None or query_embeddings.size == 0:
            return {"distances": [], "indices": []}

        try:
            # Ensure shape is (n_queries, dim)
            q = np.array(query_embeddings, dtype=np.float32)
            if q.ndim == 1:
                q = np.expand_dims(q, axis=0)

            # If index is empty, return empty lists
            if self.faiss_index.ntotal == 0:
                return {"distances": [], "indices": []}

            D, I = self.faiss_index.search(q, k)
            return {"distances": D.tolist(), "indices": I.tolist()}

        except Exception as e:
            print(f"⚠️  Indexer.search_by_embeddings error: {e}")
            return {"distances": [], "indices": []}

    async def query(self, query_text: str, k: int = 6) -> Dict:
        """
        Convenience method: embed the query_text and run a search.
        Returns structured results with metadata merged.
        """
        try:
            q_emb = await self.embed_texts([query_text])
            if q_emb.shape[0] == 0:
                return {"ids": [], "documents": [], "metadatas": [], "distances": []}

            res = self.search_by_embeddings(q_emb, k)
            distances = res.get("distances", [[]])[0]
            indices = res.get("indices", [[]])[0]

            ids = []
            docs = []
            metas = []
            for idx, dist in zip(indices, distances):
                if idx is None or idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                ids.append(meta.get("id", str(idx)))
                docs.append(meta.get("text", ""))
                metas.append(meta)
            return {"ids": ids, "documents": docs, "metadatas": metas, "distances": distances}
        except Exception as e:
            print(f"⚠️  Indexer.query error: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def clear(self):
        """Clear in-memory FAISS index and metadata (keeps persisted files until delete() is called)."""
        try:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
        except Exception as e:
            print(f"⚠️  Indexer.clear error: {e}")

    def delete_persisted(self):
        """Delete persisted index, metadata and config files from disk."""
        for path in (self.index_file_path, self.meta_file_path, self.config_file_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"⚠️  Could not remove {path}: {e}")
