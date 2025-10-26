# backend/core/utils/logger.py
"""
Centralized logging utility for MCP backend.
Provides rotating file and console loggers with consistent formatting.
"""

import logging
from logging.handlers import RotatingFileHandler
import sys
import os
from backend.core.utils.config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance."""
    log_dir = settings.log_dir
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

        # File handler (rotating)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{name.replace('.', '_')}.log"),
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8"
        )
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Initialize root logger early
root_logger = get_logger("backend")
root_logger.info("✅ Logger initialized successfully")
