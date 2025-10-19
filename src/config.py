"""
Configuration management for MCP Developer Server
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------------------------------------------
# 🧠 Model Configuration
# -------------------------------------------------------------------------
@dataclass
class HuggingFaceConfig:
    """Hugging Face + Qwen Model Configuration"""

    api_token: Optional[str] = None
    model_qwen: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_cache_dir: str = "./models_cache"
    max_new_tokens: int = 512
    temperature: float = 0.7
    max_retries: int = 3
    timeout: int = 30
    use_cache: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"

    def __post_init__(self):
        # Allow overrides via environment variables
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN", self.api_token)
        self.model_qwen = os.getenv("MODEL_NAME", self.model_qwen)
        self.model_cache_dir = os.getenv("MODEL_CACHE_DIR", self.model_cache_dir)


# -------------------------------------------------------------------------
# ⚙️ Server Configuration
# -------------------------------------------------------------------------
@dataclass
class ServerConfig:
    """MCP Server Configuration"""

    server_name: str = "devpilot-mcp"
    log_level: str = "INFO"
    cache_size: int = 100
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600

    def __post_init__(self):
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)


# -------------------------------------------------------------------------
# 🧩 Code Analysis Configuration
# -------------------------------------------------------------------------
@dataclass
class AnalysisConfig:
    """Code Analysis Configuration"""

    max_file_size: int = 100_000
    max_lines_per_function: int = 50
    max_complexity_score: float = 10.0
    enabled_checks: list[str] = field(
        default_factory=lambda: ["security", "complexity", "style", "documentation", "testing"]
    )


# -------------------------------------------------------------------------
# 🧾 Main Config Wrapper
# -------------------------------------------------------------------------
class Config:
    """Main configuration class"""

    def __init__(self):
        self.huggingface = HuggingFaceConfig()
        self.server = ServerConfig()
        self.analysis = AnalysisConfig()

    @classmethod
    def from_env(cls) -> "Config":
        """Create config instance from environment variables"""
        return cls()

    def validate(self) -> bool:
        """Validate essential config parameters"""
        if not self.huggingface.api_token:
            print("⚠️  Warning: Hugging Face API token not set (will use public access).")
        if not os.path.exists(self.huggingface.model_cache_dir):
            os.makedirs(self.huggingface.model_cache_dir, exist_ok=True)
        return True


# Global config instance
config = Config.from_env()
