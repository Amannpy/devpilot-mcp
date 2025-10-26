"""
CodePreprocessor — extracts, cleans, and chunks code files for indexing.
Supports Python, JS, Java, C++, and more.
"""

import os
import re
from typing import List, Dict
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class CodePreprocessor:
    """Extracts and normalizes code for RAG ingestion."""

    def __init__(self):
        self.supported_extensions = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".cs": "C#",
        }

    def extract_code_chunks(self, file_path: str) -> List[Dict]:
        """Reads and chunks a code file."""
        _, ext = os.path.splitext(file_path)
        if ext not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {file_path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()

            # Remove comments and excessive whitespace
            cleaned_code = re.sub(r"(?m)#.*?$|//.*?$|/\*.*?\*/", "", code, flags=re.S)
            cleaned_code = re.sub(r"\n{2,}", "\n", cleaned_code.strip())

            # Simple chunking by function or class definitions
            chunks = re.split(r"(?=def |class )", cleaned_code)
            processed_chunks = [
                {
                    "content": chunk.strip(),
                    "metadata": {
                        "file": os.path.basename(file_path),
                        "language": self.supported_extensions[ext],
                    },
                    "source": file_path,
                }
                for chunk in chunks
                if len(chunk.strip()) > 10
            ]

            logger.info(f"Extracted {len(processed_chunks)} code chunks from {file_path}")
            return processed_chunks

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
