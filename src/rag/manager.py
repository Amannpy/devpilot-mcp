"""
RAG Manager with preprocessing, confidence scoring, and fallback.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.rag.indexer import Indexer
from src.rag.retriever import Retriever
from src.rag.ingest.pdf import PDFPreprocessor
from src.rag.ingest.code import CodePreprocessor
from src.rag.ingest.text import TextPreprocessor
from src.models import model_manager

logger = logging.getLogger(__name__)


class RAGManager:
    """
    Manages the full RAG pipeline with preprocessing and confidence scoring.
    """

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        persist_dir: Optional[str] = None,
        confidence_threshold: float = 0.35,
        debug_mode: bool = False
    ):
        self.persist_dir = persist_dir or "./faiss_index"
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode or os.getenv("RAG_DEBUG", "").lower() == "true"

        self.max_chunks_in_prompt = 4
        self.embedding_dim = embedding_dim
        self._initialized = False

        # Components
        self.indexer: Optional[Indexer] = None
        self.retriever: Optional[Retriever] = None

        # Preprocessors
        self.pdf_preprocessor = PDFPreprocessor()
        self.code_preprocessor = CodePreprocessor()
        self.text_preprocessor = TextPreprocessor()

        # Debug logging
        if self.debug_mode:
            self._setup_debug_logging()

    def _setup_debug_logging(self):
        """Setup debug logging for RAG operations."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        self.debug_log_path = log_dir / "rag_debug.log"

        file_handler = logging.FileHandler(self.debug_log_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    async def _initialize(self):
        """Lazy initialization of indexer and retriever."""
        if self._initialized:
            return

        # Get embedding dimension if not set
        if self.embedding_dim is None:
            try:
                emb = await model_manager.get_code_embeddings("test")
                self.embedding_dim = len(emb) if emb else 768
            except Exception:
                self.embedding_dim = 768

        self.indexer = Indexer(
            embedding_dim=self.embedding_dim,
            persist_dir=self.persist_dir
        )

        self.retriever = Retriever(
            indexer=self.indexer,
            k_dense=6,
            k_sparse=6,
            weight_dense=0.8
        )

        self._initialized = True
        logger.info(f"✅ RAG Manager initialized (embedding_dim={self.embedding_dim})")

    async def index_file(self, file_path: str, language: Optional[str] = None):
        """
        Index a single file with preprocessing.
        """
        await self._initialize()

        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            logger.warning(f"File not found: {file_path}")
            return

        logger.info(f"📄 Indexing file: {file_path}")

        try:
            # Determine file type and preprocess
            if file_path.lower().endswith('.pdf'):
                chunks = self.pdf_preprocessor.extract_and_clean(file_path)
                logger.info(f"  → Extracted {len(chunks)} PDF chunks")

            elif file_path.lower().endswith('.py'):
                text = file_path_obj.read_text(encoding="utf-8", errors="ignore")
                chunks = self.code_preprocessor.extract_python_chunks(text, file_path)
                logger.info(f"  → Extracted {len(chunks)} Python code chunks")

            elif file_path.lower().endswith('.md'):
                text = file_path_obj.read_text(encoding="utf-8", errors="ignore")
                chunks = self.text_preprocessor.process_markdown(text, file_path)
                logger.info(f"  → Extracted {len(chunks)} Markdown chunks")

            elif file_path.lower().endswith('.txt'):
                text = file_path_obj.read_text(encoding="utf-8", errors="ignore")
                chunks = self.text_preprocessor.process_plain_text(text, file_path)
                logger.info(f"  → Extracted {len(chunks)} text chunks")

            else:
                # Fallback: treat as plain text
                text = file_path_obj.read_text(encoding="utf-8", errors="ignore")
                chunks = self.text_preprocessor.process_plain_text(text, file_path)
                logger.info(f"  → Extracted {len(chunks)} chunks (fallback)")

            if not chunks:
                logger.warning(f"  ⚠️  No valid chunks extracted from {file_path}")
                return

            # Add unique IDs to chunks
            import uuid
            for chunk in chunks:
                chunk.setdefault("id", str(uuid.uuid4()))

            # Index the chunks
            await self.indexer.add_chunks(chunks)
            self.indexer.persist()

            logger.info(f"  ✅ Indexed {len(chunks)} chunks from {file_path}")

        except Exception as e:
            logger.error(f"  ❌ Error indexing {file_path}: {e}")

    async def index_repo(
        self,
        root_path: str,
        extensions: Optional[List[str]] = None
    ) -> Dict:
        """
        Index entire repository with preprocessing.
        """
        await self._initialize()

        extensions = extensions or ['.py', '.md', '.txt', '.js', '.ts']
        files_indexed = 0
        chunks_total = 0

        logger.info(f"📂 Indexing repository: {root_path}")

        root = Path(root_path)

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue

            if not self.code_preprocessor.should_process_file(file_path):
                continue

            if file_path.suffix.lower() in extensions:
                try:
                    chunks_before = len(self.indexer.metadata)
                    await self.index_file(str(file_path))
                    chunks_after = len(self.indexer.metadata)

                    chunks_added = chunks_after - chunks_before
                    if chunks_added > 0:
                        files_indexed += 1
                        chunks_total += chunks_added

                except Exception as e:
                    logger.warning(f"  ⚠️  Skipped {file_path}: {e}")
                    continue

        self.indexer.persist()

        result = {
            "indexed_files": files_indexed,
            "total_chunks": chunks_total
        }

        logger.info(f"✅ Repository indexed: {files_indexed} files, {chunks_total} chunks")
        return result

    def _compute_retrieval_confidence(self, retrieved: List[Dict]) -> float:
        """
        Compute average confidence score from retrieved results.
        Returns value between 0 and 1.
        """
        if not retrieved:
            return 0.0

        scores = [r.get("score", 0.0) for r in retrieved]
        avg_score = sum(scores) / len(scores)

        distances = [r.get("distance", 999) for r in retrieved]
        avg_distance = sum(distances) / len(distances)

        # Lower distance → higher confidence
        distance_confidence = 1.0 / (1.0 + avg_distance)
        confidence = 0.6 * avg_score + 0.4 * distance_confidence

        return min(max(confidence, 0.0), 1.0)

    def _log_retrieved_chunks(self, retrieved: List[Dict], confidence: float):
        """Log retrieved chunks for debugging."""
        if not self.debug_mode:
            return

        logger.debug("=" * 80)
        logger.debug(f"Retrieved {len(retrieved)} chunks | Confidence: {confidence:.3f}")
        logger.debug("=" * 80)

        for idx, r in enumerate(retrieved[:3], 1):
            metadata = r.get("metadata", {})
            text = r.get("document", "")[:250]
            logger.debug(f"\nChunk {idx}:")
            logger.debug(f"  Source: {metadata.get('source', 'dense')}")
            logger.debug(f"  Score: {r.get('score', 0):.4f}")
            logger.debug(f"  Distance: {r.get('distance', 0):.4f}")
            logger.debug(f"  File: {metadata.get('file_path', 'unknown')}")
            logger.debug(f"  Type: {metadata.get('type', 'unknown')}")
            logger.debug(f"  Text preview: {text}...")
            logger.debug("-" * 80)

    def _build_prompt(
        self,
        user_query: str,
        retrieved: List[Dict],
        task_type: str,
        use_context: bool = True
    ) -> str:
        """Build prompt with task-specific instructions."""
        if not use_context or not retrieved:
            return self._build_direct_prompt(user_query, task_type)

        ctx_parts = []
        for idx, r in enumerate(retrieved[:self.max_chunks_in_prompt], 1):
            md = r.get("metadata", {})
            snippet = r.get("document", "").strip()
            file_name = Path(md.get('file_path', 'unknown')).name
            chunk_type = md.get('type', 'text')
            header = f"[Source {idx}: {file_name} | {chunk_type}]"
            ctx_parts.append(f"{header}\n{snippet}\n")

        context_text = "\n---\n".join(ctx_parts)

        prompts = {
            "review": (
                "You are an expert code reviewer. Review the code based on the context provided.\n"
                "Focus on: bugs, security issues, code smells, and best practices.\n"
            ),
            "docs": (
                "You are a technical writer. Generate clear documentation based on the context.\n"
                "Include: purpose, parameters, return values, and usage examples.\n"
            ),
            "tests": (
                "You are a testing expert. Generate pytest unit tests based on the context.\n"
                "Include: test cases for normal flow, edge cases, and error handling.\n"
            ),
            "general": (
                "Answer the user's question based on the provided context.\n"
                "Be concise, accurate, and cite specific sources when relevant.\n"
            )
        }

        instruction = prompts.get(task_type, prompts["general"])

        return (
            f"{instruction}\n"
            f"Context:\n{context_text}\n"
            f"---\n\n"
            f"User Query: {user_query}\n\n"
            f"Response:"
        )

    def _build_direct_prompt(self, user_query: str, task_type: str) -> str:
        """Build prompt without RAG context (fallback mode)."""
        prompts = {
            "review": f"Review the following:\n\n{user_query}\n\nReview:",
            "docs": f"Generate documentation for:\n\n{user_query}\n\nDocumentation:",
            "tests": f"Generate tests for:\n\n{user_query}\n\nTests:",
            "general": f"{user_query}"
        }

        return prompts.get(task_type, user_query)

    async def retrieve_and_generate(
        self,
        user_query: str,
        task: str = "general",
        force_rag: bool = False
    ) -> Dict:
        """
        Main RAG pipeline with confidence-based fallback.
        """
        await self._initialize()

        if not self.indexer.metadata:
            logger.warning("⚠️  RAG index is empty - using direct mode")
            prompt = self._build_direct_prompt(user_query, task)
            generated = await model_manager.qwen.generate_text(
                prompt, task_type=task, max_new_tokens=512
            )
            return {
                "prompt": prompt,
                "generated_text": generated,
                "retrieved": [],
                "confidence": 0.0,
                "mode": "direct (empty index)"
            }

        retrieved = await self.retriever.hybrid_retrieve(user_query, k=6)
        confidence = self._compute_retrieval_confidence(retrieved)
        self._log_retrieved_chunks(retrieved, confidence)

        use_rag = force_rag or confidence >= self.confidence_threshold
        if not use_rag:
            logger.info(
                f"⚠️  Low retrieval confidence ({confidence:.3f} < {self.confidence_threshold}) "
                f"- falling back to direct mode"
            )

        prompt = self._build_prompt(user_query, retrieved, task, use_context=use_rag)
        generated = await model_manager.qwen.generate_text(
            prompt, task_type=task, max_new_tokens=512
        )

        mode = (
            f"RAG (confidence: {confidence:.3f})"
            if use_rag
            else f"direct (low confidence: {confidence:.3f})"
        )
        logger.info(f"✅ Generated response using {mode}")

        return {
            "prompt": prompt,
            "generated_text": generated,
            "retrieved": retrieved[:self.max_chunks_in_prompt],
            "confidence": confidence,
            "mode": mode,
            "used_rag": use_rag
        }
