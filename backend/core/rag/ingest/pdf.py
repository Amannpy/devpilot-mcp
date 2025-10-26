"""
PDFPreprocessor — extract and clean text from PDF files for RAG ingestion.
"""

import os
from typing import List, Dict
from PyPDF2 import PdfReader
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class PDFPreprocessor:
    """Handles PDF text extraction and normalization."""

    def extract_and_clean(self, file_path: str) -> List[Dict]:
        """Extracts text from PDF and splits into logical chunks."""
        if not file_path.lower().endswith(".pdf"):
            logger.warning(f"Skipping non-PDF file: {file_path}")
            return []

        try:
            reader = PdfReader(file_path)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n".join(pages_text).strip()

            # Split text into chunks (~1000 chars)
            chunk_size = 1000
            chunks = [
                {
                    "content": full_text[i:i + chunk_size],
                    "metadata": {"file": os.path.basename(file_path), "page_range": f"{i // chunk_size + 1}"},
                    "source": file_path,
                }
                for i in range(0, len(full_text), chunk_size)
                if len(full_text[i:i + chunk_size].strip()) > 50
            ]

            logger.info(f"Extracted {len(chunks)} text chunks from PDF: {file_path}")
            return chunks

        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            return []
