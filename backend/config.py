"""
Backend configuration for MCP AI Server
Unified configuration - import as 'from backend.config import settings'
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AppConfig:
    """Application configuration class - single source of truth"""

    # ============================================
    # PROJECT PATHS
    # ============================================
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_CACHE_DIR = BASE_DIR / "models_cache"
    FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
    RAG_DB_DIR = BASE_DIR / "rag_db"
    CACHE_DIR = BASE_DIR / "cache"

    # ============================================
    # SERVER SETTINGS
    # ============================================
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "True").lower() == "true"

    # API Settings
    API_TITLE: str = "MCP AI Server"
    API_VERSION: str = "1.0.0"

    # ============================================
    # MODEL SETTINGS
    # ============================================
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    default_model: str = MODEL_NAME  # Alias for backwards compatibility

    MODEL_PATH: Optional[str] = os.getenv("MODEL_PATH")
    MODEL_USE_LOCAL: bool = os.getenv("MODEL_USE_LOCAL", "True").lower() == "true"
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cuda")
    MODEL_MAX_LENGTH: int = int(os.getenv("MODEL_MAX_LENGTH", "8192"))
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    MODEL_TOP_P: float = float(os.getenv("MODEL_TOP_P", "0.9"))
    MODEL_TOP_K: int = int(os.getenv("MODEL_TOP_K", "50"))

    max_tokens: int = 512  # Alias for backwards compatibility

    # Model Download Settings
    MODEL_DOWNLOAD_DIR: Path = MODELS_CACHE_DIR / "llm"
    MODEL_TRUST_REMOTE_CODE: bool = os.getenv("MODEL_TRUST_REMOTE_CODE", "True").lower() == "true"
    MODEL_OFFLOAD_FOLDER: Path = MODELS_CACHE_DIR / "offload"

    # GPU Optimization
    MODEL_LOAD_IN_8BIT: bool = os.getenv("MODEL_LOAD_IN_8BIT", "False").lower() == "true"
    MODEL_LOAD_IN_4BIT: bool = os.getenv("MODEL_LOAD_IN_4BIT", "False").lower() == "true"
    MODEL_USE_FLASH_ATTENTION: bool = os.getenv("MODEL_USE_FLASH_ATTENTION", "False").lower() == "true"
    MODEL_MAX_MEMORY: Optional[str] = os.getenv("MODEL_MAX_MEMORY", "3.5GB")

    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_model: str = EMBEDDING_MODEL  # Alias
    EMBEDDING_PATH: Optional[str] = os.getenv("EMBEDDING_PATH")
    EMBEDDING_USE_LOCAL: bool = os.getenv("EMBEDDING_USE_LOCAL", "True").lower() == "true"
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cuda")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    EMBEDDING_DIM: int = EMBEDDING_DIMENSION  # Alias
    EMBEDDING_DOWNLOAD_DIR: Path = MODELS_CACHE_DIR / "embeddings"

    # ============================================
    # RAG SETTINGS
    # ============================================
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "True").lower() == "true"
    use_rag: bool = RAG_ENABLED  # Alias
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))
    RAG_VECTOR_STORE: str = os.getenv("RAG_VECTOR_STORE", "faiss")
    RAG_INDEX_PATH: Path = FAISS_INDEX_DIR / "index"

    # ============================================
    # LOGGING SETTINGS
    # ============================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "True").lower() == "true"
    LOG_FILE_MAX_BYTES: int = int(os.getenv("LOG_FILE_MAX_BYTES", "10485760"))
    LOG_FILE_BACKUP_COUNT: int = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

    # ============================================
    # SESSION & CACHE SETTINGS
    # ============================================
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))
    session_timeout: int = SESSION_TIMEOUT  # Alias
    MAX_HISTORY_LENGTH: int = int(os.getenv("MAX_HISTORY_LENGTH", "50"))
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"

    # ============================================
    # METHODS
    # ============================================
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.DATA_DIR,
            cls.LOGS_DIR,
            cls.MODELS_CACHE_DIR,
            cls.MODEL_DOWNLOAD_DIR,
            cls.MODEL_OFFLOAD_FOLDER,
            cls.EMBEDDING_DOWNLOAD_DIR,
            cls.FAISS_INDEX_DIR,
            cls.RAG_DB_DIR,
            cls.CACHE_DIR,
            cls.RAG_INDEX_PATH.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def is_cuda_available(cls) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @classmethod
    def get_device(cls) -> str:
        """Get the appropriate device for model inference"""
        if cls.MODEL_DEVICE == "cuda" and cls.is_cuda_available():
            return "cuda"
        elif cls.MODEL_DEVICE == "mps":
            try:
                import torch
                if torch.backends.mps.is_available():
                    return "mps"
            except (ImportError, AttributeError):
                pass
        return "cpu"

    @classmethod
    def get_model_path(cls) -> str:
        """Get the path to the model (local or HuggingFace Hub)"""
        if cls.MODEL_USE_LOCAL and cls.MODEL_PATH:
            return cls.MODEL_PATH
        elif cls.MODEL_USE_LOCAL:
            local_path = cls.MODEL_DOWNLOAD_DIR / cls.MODEL_NAME.replace("/", "--")
            if local_path.exists():
                return str(local_path)
        return cls.MODEL_NAME

    @classmethod
    def get_embedding_path(cls) -> str:
        """Get the path to the embedding model"""
        if cls.EMBEDDING_USE_LOCAL and cls.EMBEDDING_PATH:
            return cls.EMBEDDING_PATH
        elif cls.EMBEDDING_USE_LOCAL:
            local_path = cls.EMBEDDING_DOWNLOAD_DIR / cls.EMBEDDING_MODEL.replace("/", "--")
            if local_path.exists():
                return str(local_path)
        return cls.EMBEDDING_MODEL

    @classmethod
    def get_gpu_info(cls) -> dict:
        """Get GPU information"""
        if not cls.is_cuda_available():
            return {"available": False}

        try:
            import torch
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
                "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count()
            }
        except Exception as e:
            return {"available": True, "error": str(e)}


# Initialize directories
AppConfig.ensure_directories()

# Create singleton instance
settings = AppConfig()

# Backwards compatibility aliases
app_config = settings
config = settings