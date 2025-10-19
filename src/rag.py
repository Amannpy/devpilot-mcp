# src/rag.py
"""
Lightweight Multi-Vector + Dynamic RAG subsystem for DevPilot MCP.

- Chunking (AST-based for Python, fallback sliding-window)
- Indexing into Chroma (persisted)
- Hybrid retrieval: dense (vector) + sparse (keyword) combination
- Prompt assembly and orchestration with existing ModelManager

Usage:
    from src.rag import RAGManager, Indexer
    rag = RAGManager()
    await rag.index_repo("/path/to/repo")
    response = await rag.retrieve_and_generate("How to improve auth flow?", task="review", language="python")
"""

import os
import uuid
import json
import asyncio
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings

from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound

from src.models import model_manager  # uses existing ModelManager singleton
from src.config import config

import ast
import re

# ---------------------------
# Utility: simple tokenizer for sparse matching
# ---------------------------
WORD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]+\b")

def simple_tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())

# ---------------------------
# Chunking
# ---------------------------
def extract_python_chunks(code: str, filename: str) -> List[Dict]:
    """
    Extracts functions, classes, and module docstrings from Python source using ast.
    Returns list of dicts with keys: id, text, file_path, start_line, end_line, type, language
    """
    chunks = []
    try:
        tree = ast.parse(code)
        source_lines = code.splitlines()
        # module docstring
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
                end = getattr(node, "end_lineno", None)
                if end is None:
                    # best-effort fallback: try to find until next definition
                    end = start + 1
                # slice source lines defensively
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
        # fallback to single chunk
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
    """
    Fallback chunker: sliding window on characters, conservative sizes.
    """
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
# Indexer: Chroma + embedding via model_manager
# ---------------------------
class Indexer:
    def __init__(self, collection_name: str = "devpilot", persist_dir: Optional[str] = None):
        persist_dir = persist_dir or getattr(config.huggingface, "model_cache_dir", "./models_cache")
        self.persist_dir = os.path.join(persist_dir, "chroma_db")
        os.makedirs(self.persist_dir, exist_ok=True)
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_dir)
        self.client = chromadb.Client(settings)
        # create or get
        try:
            self.col = self.client.get_collection(collection_name)
        except Exception:
            self.col = self.client.create_collection(collection_name)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Use model_manager to create embeddings for a list of texts.
        model_manager.get_code_embeddings is async and returns List[float] for a single text.
        We call it sequentially to avoid parallel GPU memory overhead; can be batched later.
        """
        embeddings = []
        for t in texts:
            emb = await model_manager.get_code_embeddings(t)
            if emb is None:
                # fallback: small zero-vector (avoid crash)
                embeddings.append([0.0])
            else:
                embeddings.append(emb)
        return embeddings

    async def add_chunks(self, chunks: List[Dict]):
        """
        Adds chunks to chroma: chunks is list of metadata dicts with 'text' and metadata.
        """
        docs = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [{k: v for k, v in c.items() if k not in ("text", "id")} for c in chunks]

        embeddings = await self.embed_texts(docs)
        # Chroma expects embeddings as list of lists (floats)
        # it can accept different dims; store them as-is
        self.col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    def persist(self):
        try:
            self.client.persist()
        except Exception:
            # some chroma impls persist automatically
            pass

    def delete_collection(self):
        try:
            self.client.delete_collection(self.col.name)
        except Exception:
            pass

# ---------------------------
# Retriever: hybrid dense + sparse
# ---------------------------
class Retriever:
    def __init__(self, indexer: Indexer, k_dense: int = 6, k_sparse: int = 6, weight_dense: float = 0.8):
        self.indexer = indexer
        self.k_dense = k_dense
        self.k_sparse = k_sparse
        self.weight_dense = weight_dense

    async def dense_query(self, query: str, k: int = None, where: Optional[dict]=None):
        k = k or self.k_dense
        # embed query
        q_embs = await self.indexer.embed_texts([query])
        try:
            res = self.indexer.col.query(query_embeddings=q_embs, n_results=k, where=where or {})
        except Exception as e:
            # fallback: empty result structure
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}
        return res

    def sparse_query(self, query: str, k: int = None, where: Optional[dict]=None):
        """
        Keyword-based sparse retrieval: score by keyword overlap.
        This is a lightweight fallback for lexical matching (multi-vector idea).
        """
        k = k or self.k_sparse
        tokens = set(simple_tokenize(query))
        # retrieve some candidates from chroma (no embedding)
        # since Chroma doesn't have an efficient text-only search here, we will fetch a batch
        # NOTE: for large datasets, replace this with a real sparse index
        candidates = self.indexer.col.get(include=["ids","metadatas","documents"], limit=500)
        scored = []
        for i, doc in enumerate(candidates.get("documents", [])):
            meta = candidates.get("metadatas", [])[i] if candidates.get("metadatas") else {}
            if where:
                # simple metadata filter
                ok = True
                for kf, vf in (where or {}).items():
                    if meta.get(kf) != vf:
                        ok = False
                        break
                if not ok:
                    continue
            doc_tokens = set(simple_tokenize(doc))
            overlap = len(tokens & doc_tokens)
            if overlap > 0:
                scored.append((overlap, candidates["ids"][i], doc, meta))
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:k]
        ids = [t[1] for t in top]
        docs = [t[2] for t in top]
        metas = [t[3] for t in top]
        # sparse distances: inverse of overlap (higher overlap -> lower distance)
        distances = [1.0 / (1 + t[0]) for t in top]
        return {"ids": ids, "documents": docs, "metadatas": metas, "distances": distances}

    async def hybrid_retrieve(self, query: str, k: int = 6, language: Optional[str]=None):
        """
        Combine dense + sparse results, dedupe and rank by a simple weighted score.
        Returns a list of dicts: [{id, document, metadata, distance, score}]
        """
        where = {"language": language} if language else None
        dense = await self.dense_query(query, k=k, where=where)
        sparse = self.sparse_query(query, k=k, where=where)

        # collect candidates
        candidates = {}
        # dense: distances lower => better (chroma returns distances; assume smaller better)
        dense_docs = dense.get("documents", [])
        dense_ids = dense.get("ids", [])
        dense_dists = dense.get("distances", [])
        for i, _id in enumerate(dense_ids):
            doc = dense_docs[i]
            dist = dense_dists[i] if i < len(dense_dists) else 1.0
            # convert distance to score (higher better)
            score = (1.0 / (1.0 + dist)) * self.weight_dense
            candidates[_id] = {"id": _id, "document": doc, "metadata": dense.get("metadatas", [{}])[i] if dense.get("metadatas") else {}, "score": score, "source": "dense"}

        # sparse
        for i, _id in enumerate(sparse.get("ids", [])):
            if _id in candidates:
                # boost existing candidate
                candidates[_id]["score"] +=  (1.0 / (1.0 + sparse["distances"][i])) * (1.0 - self.weight_dense)
            else:
                candidates[_id] = {"id": _id, "document": sparse["documents"][i], "metadata": sparse["metadatas"][i], "score": (1.0 / (1.0 + sparse["distances"][i])) * (1.0 - self.weight_dense), "source": "sparse"}

        # produce sorted list
        ranked = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:k]

# ---------------------------
# RAG Manager: prompt builder + generation call
# ---------------------------
class RAGManager:
    def __init__(self, collection_name: str = "devpilot", persist_dir: Optional[str] = None):
        self.indexer = Indexer(collection_name=collection_name, persist_dir=persist_dir)
        self.retriever = Retriever(self.indexer)
        # prompt size / number of chunks to include
        self.max_chunks_in_prompt = 4

    async def index_file(self, file_path: str, language: Optional[str] = None):
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        lang = language or self.detect_language(file_path, text)
        if lang == "python":
            chunks = extract_python_chunks(text, file_path)
        else:
            chunks = sliding_window_chunks(text, file_path, language=lang)
        # attach inferred language
        for c in chunks:
            c.setdefault("language", lang)
        await self.indexer.add_chunks(chunks)
        self.indexer.persist()

    async def index_repo(self, root_path: str, extensions: Optional[List[str]] = None):
        """
        Walk a repository directory and index files.
        """
        extensions = extensions or ['.py', '.md', '.txt', '.js', '.ts', '.java']
        root = Path(root_path)
        files_indexed = 0
        for p in root.rglob("*"):
            if p.is_file():
                if any(part in p.parts for part in ('.git', 'node_modules', 'venv', '__pycache__')):
                    continue
                if p.suffix.lower() in extensions:
                    try:
                        await self.index_file(str(p))
                        files_indexed += 1
                    except Exception:
                        # skip problematic files
                        continue
        self.indexer.persist()
        return {"indexed_files": files_indexed, "collection": self.indexer.col.name}

    def detect_language(self, filename: str, text: str) -> str:
        """
        Simple language detection: by extension first, fallback to pygments guess.
        """
        ext = Path(filename).suffix.lower()
        if ext in ('.py',):
            return "python"
        if ext in ('.js', '.jsx', '.ts', '.tsx'):
            return "javascript"
        try:
            lexer = guess_lexer_for_filename(filename, text)
            return lexer.name.lower()
        except ClassNotFound:
            return "text"

    def _build_prompt(self, user_query: str, retrieved: List[Dict], task_type: str) -> str:
        """
        Build a prompt that places retrieved context before the user query.
        Keep it concise and deterministic.
        """
        ctx_parts = []
        for r in retrieved[: self.max_chunks_in_prompt]:
            md = r.get("metadata", {})
            header = f"File: {md.get('file_path', 'unknown')} | Type: {md.get('type','?')}"
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
        full = f"{instruction}\n\nContext:\n{ctx_text}\n\nUser Query:\n{user_query}\n\nResponse:\n"
        return full

    async def retrieve_and_generate(self, user_query: str, task: str = "review", language: Optional[str] = None, k:int=6) -> Dict:
        """
        Main integration point: retrieve hybrid results, build prompt, and call ModelManager for generation.
        Returns dict with 'prompt', 'generated_text', 'retrieved' (list of metadata+snippet)
        """
        retrieved = await self.retriever.hybrid_retrieve(user_query, k=k, language=language)
        prompt = self._build_prompt(user_query, retrieved, task)
        # call existing model_manager for generation; use embeddings=None (we pass prompt)
        generated = await model_manager.qwen.generate_text(prompt, task_type=task, embeddings=None, max_new_tokens=512)
        return {
            "prompt": prompt,
            "generated_text": generated,
            "retrieved": retrieved
        }
