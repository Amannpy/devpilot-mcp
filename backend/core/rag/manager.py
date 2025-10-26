from backend.core.rag.indexer import Indexer
from backend.core.rag.retriever import Retriever
from backend.core.rag.ingest.code import CodePreprocessor
from backend.core.rag.ingest.pdf import PDFPreprocessor
from backend.core.rag.ingest.text import TextPreprocessor
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class RAGManager:
    """Optional RAG Manager for retrieval-augmented generation."""

    def __init__(self, embedding_dim: int = 768):
        self.indexer = Indexer(embedding_dim)
        self.retriever = Retriever(self.indexer)
        self.embedding_dim = embedding_dim

        self.code_preprocessor = CodePreprocessor()
        self.pdf_preprocessor = PDFPreprocessor()
        self.text_preprocessor = TextPreprocessor()

    async def index_file(self, file_path: str):
        path = Path(file_path)
        ext = path.suffix.lower()
        chunks: List[Dict] = []

        if ext in self.code_preprocessor.supported_extensions:
            chunks = self.code_preprocessor.extract_code_chunks(file_path)
        elif ext == ".pdf":
            chunks = self.pdf_preprocessor.extract_and_clean(file_path)
        elif ext in [".txt", ".md", ".rst"]:
            chunks = self.text_preprocessor.extract_text_chunks(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return

        if chunks:
            await self.indexer.add_chunks(chunks)
            logger.info(f"Indexed {len(chunks)} chunks from {file_path}")

    async def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        return await self.retriever.hybrid_retrieve(query, k=k)
