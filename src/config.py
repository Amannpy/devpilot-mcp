"""
Configuration management for MCP Developer Server
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class HuggingFaceConfig:
    """HuggingFace API Configuration"""
    api_token: Optional[str] = None
    model_code_bert: str = "microsoft/CodeBERT-base"
    model_flan_t5: str = "google/flan-t5-base"
    max_retries: int = 3
    timeout: int = 30
    use_cache: bool = True

    def __post_init__(self):
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN", self.api_token)


@dataclass
class ServerConfig:
    """MCP Server Configuration"""
    server_name: str = "intelligent-dev-workflow"
    log_level: str = "INFO"
    cache_size: int = 100
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600

    def __post_init__(self):
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)

@dataclass
class AnalysisConfig:
    """Code Analysis Configuration"""
    max_file_size: int = 100000
    max_lines_per_function: int = 50
    max_complexity_score: float = 10.0
    enabled_checks: list[str] = None

    def __post_init__(self):
        if self.enabled_checks is None:
            self.enabled_checks = [
                "security", "complexity", "style",
                "documentation", "testing"
            ]

class Config:
    """Main configuration class"""
    def __init__(self):
        self.huggingface = HuggingFaceConfig()
        self.server = ServerConfig()
        self.analysis = AnalysisConfig()

    @classmethod
    def from_env(cls) -> "Config":
        return cls()

    def validate(self) -> bool:
        if not self.huggingface.api_token:
            print("Warning: HuggingFace API token not set.")
        return True

config = Config.from_env()