"""
Service for smart prompt construction and intent detection.
Helps the model understand *what the user really wants* (e.g., explain, refactor, debug, document).
"""

import re
from enum import Enum
from typing import Optional
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)


class Intent(str, Enum):
    EXPLAIN = "explain"
    OPTIMIZE = "optimize"
    DEBUG = "debug"
    DOCUMENT = "document"
    GENERAL = "general"


def detect_intent(user_prompt: str) -> Intent:
    """
    Detect user intent based on keywords and phrasing.
    """
    prompt_lower = user_prompt.lower()
    if any(k in prompt_lower for k in ["explain", "understand", "describe"]):
        return Intent.EXPLAIN
    if any(k in prompt_lower for k in ["optimize", "improve", "enhance", "refactor"]):
        return Intent.OPTIMIZE
    if any(k in prompt_lower for k in ["bug", "error", "issue", "fix"]):
        return Intent.DEBUG
    if any(k in prompt_lower for k in ["doc", "documentation", "comment"]):
        return Intent.DOCUMENT
    return Intent.GENERAL


def build_prompt(
    user_prompt: str,
    code: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """
    Build a structured and contextual prompt for the model.
    Combines user intent, code snippet, and optional PR or RAG context.
    """
    intent = detect_intent(user_prompt)

    logger.info(f"🧠 Detected intent '{intent.value}' for prompt: {user_prompt[:50]}...")

    base_instruction = f"The user wants to {intent.value} the given code."

    prompt_parts = [base_instruction]

    if code:
        prompt_parts.append(f"\nCode:\n{code.strip()}\n")

    if context:
        prompt_parts.append(f"\nContext / PR description:\n{context.strip()}\n")

    prompt_parts.append(f"\nUser query:\n{user_prompt}\n")
    prompt_parts.append("Respond clearly and comprehensively, using best practices when applicable.")

    full_prompt = "\n".join(prompt_parts)
    logger.debug(f"Built prompt (truncated to 200 chars): {full_prompt[:200]}...")

    return full_prompt
