# src/rag_faiss.py
"""
FAISS-based Lightweight Multi-Vector + Dynamic RAG subsystem for DevPilot MCP.

- Chunking (AST-based for Python, fallback sliding-window)
- Indexing into FAISS + local metadata store (pickle/JSON)
- Hybrid retrieval: dense (vector) + sparse (keyword) combination
- Prompt assembly and orchestration with existing ModelManager
"""

import os
import uuid
import json
import asyncio
from typing import List, Dict, Optional
from pathlib import Path
import pickle
import numpy as np
import ast
import re

from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound

from src.models import model_manager  # existing ModelManager singleton

# ---------------------------
# Simple tokenizer for sparse matching
# ---------------------------
WORD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]+\b")

def simple_tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

# ---------------------------
# Chunking
# ---------------------------
def extract_python_chunks(code: str, filename: str) -> List[Dict]:
    chunks = []
    try:
        tree = ast.parse(code)
        source_lines = code.splitlines()
        module_doc = ast.get_docstring(tree)
        if module_doc:
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": module_doc,
                "file_path": filename,
                "start_line": 1,
                "end_line": 1,
                "type": "docstring",
                "language": "python"
            })
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(node, "lineno", 1) - 1
                end = getattr(node, "end_lineno", start + 1)
                snippet = "\n".join(source_lines[start:end])
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": snippet,
                    "file_path": filename,
                    "start_line": start + 1,
                    "end_line": end,
                    "type": kind,
                    "language": "python"
                })
    except SyntaxError:
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": code,
            "file_path": filename,
            "start_line": 1,
            "end_line": len(code.splitlines()),
            "type": "file",
            "language": "python"
        })
    return chunks

def sliding_window_chunks(text: str, file_path: str, language: str="text", max_chars: int=2000, overlap: int=200) -> List[Dict]:
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + max_chars, L)
        snippet = text[i:end]
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": snippet,
            "file_path": file_path,
            "start_char": i,
            "end_char": end,
            "type": "file",
            "language": language
        })
        if end == L:
            break
        i = end - overlap
    return chunks

# ---------------------------
# Indexer using FAISS + metadata
# ---------------------------
import faiss

class Indexer:
    def __init__(self, embedding_dim: int = 768, persist_dir: Optional[str] = None):
        self.embedding_dim = embedding_dim
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)  # CPU-only
        self.metadata: List[Dict] = []
        self.persist_dir = persist_dir or "./faiss_index"
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_file_path = os.path.join(self.persist_dir, "index.faiss")
        self.meta_file_path = os.path.join(self.persist_dir, "metadata.pkl")

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for t in texts:
            emb = await model_manager.get_code_embeddings(t)  # returns List[float]
            if emb is None:
                emb = np.zeros(self.embedding_dim, dtype=np.float32)
            embeddings.append(np.array(emb, dtype=np.float32))
        return np.stack(embeddings, axis=0)

    async def add_chunks(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        embeddings = await self.embed_texts(texts)
        self.faiss_index.add(embeddings)
        # store metadata alongside embedding order
        self.metadata.extend(chunks)

    def persist(self):
        faiss.write_index(self.faiss_index, self.index_file_path)
        with open(self.meta_file_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(self.index_file_path):
            self.faiss_index = faiss.read_index(self.index_file_path)
        if os.path.exists(self.meta_file_path):
            with open(self.meta_file_path, "rb") as f:
                self.metadata = pickle.load(f)

# ---------------------------
# Retriever: FAISS dense + keyword sparse hybrid
# ---------------------------
class Retriever:
    def __init__(self, indexer: Indexer, k_dense: int = 6, k_sparse: int = 6, weight_dense: float = 0.8):
        self.indexer = indexer
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.weight_dense = weight_dense

    async def dense_query(self, query: str, k: int = None):
        k = k or self.k_dense
        q_emb = await self.indexer.embed_texts([query])
        D, I = self.indexer.faiss_index.search(q_emb, k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx >= len(self.indexer.metadata):
                continue
            results.append({
                "id": self.indexer.metadata[idx]["id"],
                "document": self.indexer.metadata[idx]["text"],
                "metadata": self.indexer.metadata[idx],
                "distance": float(dist),
                "score": 1.0 / (1.0 + dist),
                "source": "dense"
            })
        return results

    def sparse_query(self, query: str, k: int = None):
        k = k or self.k_sparse
        tokens = set(simple_tokenize(query))
        scored = []
        for meta in self.indexer.metadata:
            doc_tokens = set(simple_tokenize(meta["text"]))
            overlap = len(tokens & doc_tokens)
            if overlap > 0:
                scored.append((overlap, meta))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:k]
        results = []
        for overlap, meta in top:
            results.append({
                "id": meta["id"],
                "document": meta["text"],
                "metadata": meta,
                "distance": 1.0 / (1 + overlap),
                "score": (1.0 / (1 + overlap)) * (1 - self.weight_dense),
                "source": "sparse"
            })
        return results

    async def hybrid_retrieve(self, query: str, k: int = 6):
        dense_results = await self.dense_query(query, k)
        sparse_results = self.sparse_query(query, k)
        # merge by id and sum scores
        candidates = {r["id"]: r for r in dense_results}
        for r in sparse_results:
            if r["id"] in candidates:
                candidates[r["id"]]["score"] += r["score"]
            else:
                candidates[r["id"]] = r
        ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]

# ---------------------------
# RAG Manager
# ---------------------------
class RAGManager:
    def __init__(self, embedding_dim: int = 768, persist_dir: Optional[str] = None):
        self.indexer = Indexer(embedding_dim=embedding_dim, persist_dir=persist_dir)
        self.retriever = Retriever(self.indexer)
        self.max_chunks_in_prompt = 4

    async def index_file(self, file_path: str, language: Optional[str] = None):
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        lang = language or self.detect_language(file_path, text)
        if lang == "python":
            chunks = extract_python_chunks(text, file_path)
        else:
            chunks = sliding_window_chunks(text, file_path, language=lang)
        for c in chunks:
            c.setdefault("language", lang)
        await self.indexer.add_chunks(chunks)
        self.indexer.persist()

    async def index_repo(self, root_path: str, extensions: Optional[List[str]] = None):
        extensions = extensions or ['.py', '.md', '.txt', '.js', '.ts', '.java']
        files_indexed = 0
        for p in Path(root_path).rglob("*"):
            if p.is_file() and not any(part in p.parts for part in ('.git', 'node_modules', 'venv', '__pycache__')):
                if p.suffix.lower() in extensions:
                    try:
                        await self.index_file(str(p))
                        files_indexed += 1
                    except Exception:
                        continue
        self.indexer.persist()
        return {"indexed_files": files_indexed}

    def detect_language(self, filename: str, text: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext == ".py":
            return "python"
        if ext in ('.js', '.ts'):
            return "javascript"
        try:
            lexer = guess_lexer_for_filename(filename, text)
            return lexer.name.lower()
        except ClassNotFound:
            return "text"

    def _build_prompt(self, user_query: str, retrieved: List[Dict], task_type: str) -> str:
        ctx_parts = []
        for r in retrieved[:self.max_chunks_in_prompt]:
            md = r.get("metadata", {})
            header = f"File: {md.get('file_path','unknown')} | Type: {md.get('type','?')}"
            snippet = r.get("document","").strip()
            ctx_parts.append(f"{header}\n{snippet}\n---\n")
        ctx_text = "\n".join(ctx_parts)
        prompt_map = {
            "review": "You are an expert code reviewer. Use the context to identify bugs, code smells, and suggestions.",
            "docs": "You are a technical writer. Use the context to produce docstrings and short usage examples.",
            "tests": "You are an expert at writing unit tests (pytest). Use the context to generate test cases.",
            "general": "Explain the code using the provided context."
        }
        instruction = prompt_map.get(task_type, prompt_map["general"])
        return f"{instruction}\n\nContext:\n{ctx_text}\n\nUser Query:\n{user_query}\n\nResponse:\n"

    async def retrieve_and_generate(self, user_query: str, task: str = "review") -> Dict:
        retrieved = await self.retriever.hybrid_retrieve(user_query)
        prompt = self._build_prompt(user_query, retrieved, task)
        generated = await model_manager.qwen.generate_text(prompt, task_type=task, embeddings=None, max_new_tokens=512)
        return {
            "prompt": prompt,
            "generated_text": generated,
            "retrieved": retrieved
        }
