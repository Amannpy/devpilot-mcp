import pytest
from backend.services.prompt_service import detect_intent, build_prompt, Intent

def test_detect_intent_various_prompts():
    assert detect_intent("Explain this function") == Intent.EXPLAIN
    assert detect_intent("Optimize loop performance") == Intent.OPTIMIZE
    assert detect_intent("Fix this bug please") == Intent.DEBUG
    assert detect_intent("Add documentation") == Intent.DOCUMENT
    assert detect_intent("Tell me something cool") == Intent.GENERAL

def test_build_prompt_basic_structure():
    user_prompt = "Optimize this method for speed"
    code = "def slow_func(): pass"
    context = "Performance optimization for backend"
    prompt = build_prompt(user_prompt, code, context)

    assert "optimize" in prompt.lower()
    assert "Performance optimization" in prompt
    assert "slow_func" in prompt
    assert "Respond clearly" in prompt

def test_build_prompt_without_code_or_context():
    result = build_prompt("Explain how this works")
    assert "explain" in result.lower()
    assert "User query" in result
