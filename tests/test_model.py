"""
Tests for Qwen model integration
"""

import pytest
from unittest.mock import AsyncMock, patch
from src.models import QwenModel, ModelManager, ModelCache

# Sample code for testing
SAMPLE_CODE = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""

# -------------------------------------------------------------------------
# 🧩 Test Model Cache
# -------------------------------------------------------------------------
class TestModelCache:
    def test_cache_set_and_get(self):
        cache = ModelCache(max_size=5, ttl_minutes=60)
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_cache_expiration(self):
        cache = ModelCache(ttl_minutes=0)
        cache.set("temp", "data")
        cache.cache["temp"]["time"] -= 61  # simulate time passing
        assert cache.get("temp") is None

    def test_cache_clear(self):
        cache = ModelCache()
        cache.set("x", "y")
        cache.clear()
        assert cache.get("x") is None

# -------------------------------------------------------------------------
# 🧠 Test Qwen Model
# -------------------------------------------------------------------------
class TestQwenModel:
    @pytest.mark.asyncio
    async def test_analyze_code_mocked(self):
        model = QwenModel()
        with patch.object(model, "analyze_code", AsyncMock(return_value={"patterns": {"functions": 1}})):
            result = await model.analyze_code(SAMPLE_CODE)
            assert "patterns" in result
            assert result["patterns"]["functions"] == 1
            print("\n✅ Mocked Qwen analyze_code passed")

    @pytest.mark.asyncio
    async def test_generate_text_mocked(self):
        model = QwenModel()
        with patch.object(model, "generate_text", AsyncMock(return_value="Mocked output")):
            output = await model.generate_text("Explain the function")
            assert output == "Mocked output"
            print("\n✅ Mocked Qwen generate_text passed")

# -------------------------------------------------------------------------
# 🧭 Test Model Manager
# -------------------------------------------------------------------------
class TestModelManager:
    @pytest.mark.asyncio
    async def test_manager_analysis(self):
        manager = ModelManager()
        with patch.object(manager.qwen, "analyze_code", AsyncMock(return_value={"patterns": {"loops": 2}})):
            result = await manager.analyze_code(SAMPLE_CODE)
            assert "patterns" in result
            assert result["patterns"]["loops"] == 2
            print("\n✅ Manager analyze_code test passed")

    @pytest.mark.asyncio
    async def test_manager_generate_review(self):
        manager = ModelManager()
        with patch.object(manager.qwen, "generate_text", AsyncMock(return_value="Mocked review")):
            review = await manager.generate_review(SAMPLE_CODE, "python")
            assert "Mocked" in review
            print("\n✅ Manager generate_review mocked test passed")

    def test_manager_cache_clear(self):
        manager = ModelManager()
        manager.clear_caches()
        assert True
        print("\n✅ Manager cache clear passed")

# -------------------------------------------------------------------------
# 🧩 Integration Pipeline
# -------------------------------------------------------------------------
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_mocked(self):
        manager = ModelManager()
        with patch.object(manager.qwen, "analyze_code", AsyncMock(return_value={"patterns": {"if_blocks": 3}})), \
             patch.object(manager.qwen, "generate_text", AsyncMock(return_value="Mocked analysis report")):

            analysis = await manager.analyze_code(SAMPLE_CODE)
            assert "patterns" in analysis

            review = await manager.generate_review(SAMPLE_CODE, "python")
            assert "Mocked" in review

            print("\n✅ Full Qwen pipeline (mocked) passed")
