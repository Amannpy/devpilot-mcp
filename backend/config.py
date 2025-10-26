"""
Backend configuration for MCP AI Server.
Handles environment variables, API keys, paths, and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# -----------------------------
# Application Settings
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models_cache"
RAG_DB_DIR = BASE_DIR / "rag_db"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"

# -----------------------------
# FastAPI / Server Settings
# -----------------------------
API_TITLE = os.getenv("API_TITLE", "MCP AI Server")
API_VERSION = os.getenv("API_VERSION", "2.0")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# -----------------------------
# Model / RAG Settings
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "qwen_model"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "local-code-embedding")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))
USE_RAG = os.getenv("USE_RAG", "True").lower() in ("true", "1", "yes")

# -----------------------------
# Logging Settings
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "mcp_backend.log"))

# -----------------------------
# Misc
# -----------------------------
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", 3600))  # in seconds
