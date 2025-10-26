"""
TextPreprocessor — clean, normalize, and chunk plain text or Markdown files for indexing.
"""

import os
import re
from typing import List, Dict
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Handles plain text, Markdown, and RST files."""

    def extract_text_chunks(self, file_path: str) -> List[Dict]:
        """Read and chunk text-based documents."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            # Clean and normalize
            cleaned = re.sub(r"\s+", " ", text.strip())

            # Split into paragraphs or sections
            paragraphs = re.split(r"(?<=\.)\s+", cleaned)
            chunks = [
                {
                    "content": p.strip(),
                    "metadata": {"file": os.path.basename(file_path)},
                    "source": file_path,
                }
                for p in paragraphs
                if len(p.strip()) > 40
            ]

            logger.info(f"Extracted {len(chunks)} text chunks from {file_path}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return []
