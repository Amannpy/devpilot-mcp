"""
Central logging utility for MCP backend.
"""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Basic console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(console_handler)

        try:
            # Lazy import to avoid circular dependency
            from backend.core.utils.config import app_config
            log_file = Path(app_config.LOGS_DIR) / "mcp_server.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(file_handler)
        except Exception:
            # Skip file handler if config not yet initialized
            pass

        logger.info("✅ Logger initialized successfully")

    return logger
