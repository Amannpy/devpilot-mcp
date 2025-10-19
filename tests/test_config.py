"""
Unit tests for MCP Developer Server configuration
"""

import os
import shutil
import pytest
from src.config import Config, HuggingFaceConfig, ServerConfig, AnalysisConfig


@pytest.fixture(scope="function", autouse=True)
def clean_env():
    """Clean up environment variables and cache directory before each test."""
    for var in ["HUGGINGFACE_API_TOKEN", "MODEL_NAME", "MODEL_CACHE_DIR", "LOG_LEVEL"]:
        os.environ.pop(var, None)
    yield
    if os.path.exists("./models_cache_test"):
        shutil.rmtree("./models_cache_test", ignore_errors=True)


# -------------------------------------------------------------------------
# 🔧 Test HuggingFace / Qwen Config
# -------------------------------------------------------------------------
def test_huggingface_config_defaults():
    """Ensure Qwen model config loads with default values."""
    cfg = HuggingFaceConfig()
    assert "Qwen" in cfg.model_qwen
    assert cfg.model_cache_dir == "./models_cache"
    assert cfg.max_new_tokens == 512
    assert cfg.temperature == pytest.approx(0.7)
    assert cfg.use_cache is True


def test_huggingface_config_env_override():
    """Test that environment variables override default config values."""
    os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-mini"
    os.environ["MODEL_CACHE_DIR"] = "./models_cache_test"
    os.environ["HUGGINGFACE_API_TOKEN"] = "dummy-token"

    cfg = HuggingFaceConfig()
    assert cfg.model_qwen == "Qwen/Qwen2.5-mini"
    assert cfg.model_cache_dir == "./models_cache_test"
    assert cfg.api_token == "dummy-token"


# -------------------------------------------------------------------------
# ⚙️ Test Server Config
# -------------------------------------------------------------------------
def test_server_config_env_override():
    """Test server configuration with environment variable override."""
    os.environ["LOG_LEVEL"] = "DEBUG"
    cfg = ServerConfig()
    assert cfg.log_level == "DEBUG"
    assert cfg.server_name == "devpilot-mcp"


# -------------------------------------------------------------------------
# 🧩 Test Analysis Config
# -------------------------------------------------------------------------
def test_analysis_config_defaults():
    """Verify analysis configuration defaults and checks."""
    cfg = AnalysisConfig()
    assert isinstance(cfg.enabled_checks, list)
    assert "security" in cfg.enabled_checks
    assert "documentation" in cfg.enabled_checks
    assert cfg.max_file_size == 100_000


# -------------------------------------------------------------------------
# 🧾 Test Full Config Wrapper
# -------------------------------------------------------------------------
def test_full_config_initialization_and_validation(tmp_path):
    """Ensure Config initializes all sections and validates correctly."""
    cache_dir = tmp_path / "models_cache"
    os.environ["MODEL_CACHE_DIR"] = str(cache_dir)

    cfg = Config.from_env()
    assert cfg.huggingface.model_cache_dir == str(cache_dir)
    assert isinstance(cfg.server, ServerConfig)
    assert isinstance(cfg.analysis, AnalysisConfig)

    # Validate should auto-create cache directory
    result = cfg.validate()
    assert result is True
    assert os.path.exists(cfg.huggingface.model_cache_dir)
