# tests/test_config.py
from src.config import Config

def test_hf_token_loaded():
    cfg = Config()
    assert cfg.huggingface.api_token is not None, "HF token not loaded from .env"
