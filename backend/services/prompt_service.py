"""
Service for smart prompt construction and intent detection.
Helps the model understand *what the user really wants* (e.g. explain, refactor, debug, document).
"""

import re
from enum import Enum


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
    prompt = user_prompt.lower()
    if any(k in prompt for k in ["explain", "understand", "describe"]):
        return Intent.EXPLAIN
    if any(k in prompt for k in ["optimize", "improve", "enhance", "refactor"]):
        return Intent.OPTIMIZE
    if any(k in prompt for k in ["bug", "error", "issue", "fix"]):
        return Intent.DEBUG
    if any(k in prompt for k in ["doc", "documentation", "comment"]):
        return Intent.DOCUMENT
    return Intent.GENERAL


def build_prompt(user_prompt: str, code: str | None = None, context: str | None = None) -> str:
    """
    Build a more structured and contextual prompt for the model.
    """
    intent = detect_intent(user_prompt)

    base_instruction = f"User wants to {intent.value} the given code."

    prompt = f"{base_instruction}\n\n"
    if code:
        prompt += f"Code:\n{code.strip()}\n\n"
    if context:
        prompt += f"Context or PR description:\n{context.strip()}\n\n"

    prompt += f"User query: {user_prompt}\n\n"
    prompt += "Respond clearly and comprehensively, using best practices when applicable."

    return prompt
