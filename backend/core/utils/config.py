# backend/core/utils/config.py
"""
Global configuration management for MCP AI Server.
Loads environment variables and provides typed access across modules.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application-wide configuration values."""

    # --- General ---
    app_name: str = Field("MCP AI Server", description="Application name")
    environment: str = Field("development", description="Environment (development|staging|production)")
    debug: bool = Field(True, description="Run in debug mode")

    # --- Model / RAG settings ---
    model_name: str = Field("Qwen-2.5-7B-Instruct", description="Default base model name")
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model used for RAG")
    index_path: str = Field("faiss_index/index.faiss", description="FAISS index location")
    rag_db_path: str = Field("rag_db/", description="Path for RAG database or vector store")

    # --- Logging ---
    log_dir: str = Field("logs", description="Directory where log files are stored")
    log_level: str = Field("INFO", description="Logging level")

    # --- API / Security ---
    api_prefix: str = Field("/api", description="API prefix for versioning")
    secret_key: str = Field("supersecretkey", description="JWT or session secret key (placeholder)")

    # --- File I/O ---
    data_dir: str = Field("data", description="Directory containing ingested data")
    models_cache_dir: str = Field("models_cache", description="Cache for downloaded models")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance (singleton)."""
    return Settings()


# Initialize directories automatically
def ensure_directories(settings: Settings):
    """Ensure critical directories exist before app launch."""
    for path in [settings.log_dir, settings.data_dir, settings.models_cache_dir, settings.rag_db_path]:
        os.makedirs(path, exist_ok=True)


# Load settings globally on import
settings = get_settings()
ensure_directories(settings)
