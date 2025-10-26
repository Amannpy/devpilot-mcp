# backend/core/utils/file_ops.py
"""
File operations utilities for the MCP backend.
Provides safe read/write helpers for ingestion and RAG indexing.
"""

import os
import json
from pathlib import Path
from typing import Any, List, Union
from backend.core.utils.config import settings


def read_text_file(file_path: Union[str, Path]) -> str:
    """Read text content from a file safely."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def write_text_file(file_path: Union[str, Path], content: str):
    """Write text content to a file (creates directories if needed)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(file_path: Union[str, Path]) -> Any:
    """Read and parse a JSON file."""
    return json.loads(read_text_file(file_path))


def write_json(file_path: Union[str, Path], data: Any):
    """Serialize and write data to a JSON file."""
    write_text_file(file_path, json.dumps(data, indent=2))


def list_files_in_dir(directory: Union[str, Path], extensions: List[str] = None) -> List[Path]:
    """List all files in a directory with optional extension filter."""
    directory = Path(directory)
    if not directory.exists():
        return []
    files = [f for f in directory.iterdir() if f.is_file()]
    if extensions:
        files = [f for f in files if f.suffix.lower() in extensions]
    return files


def get_data_files() -> List[Path]:
    """Convenience helper to list all files in the data directory."""
    return list_files_in_dir(settings.data_dir, extensions=[".py", ".txt", ".pdf"])


def safe_remove(file_path: Union[str, Path]) -> bool:
    """Safely remove a file, ignoring errors."""
    try:
        os.remove(file_path)
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False
