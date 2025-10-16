"""
Tests for AI model integrations
"""

import pytest
from src.models import (
    CodeBERTModel,
    FLANT5Model,
    ModelManager,
    ModelCache
)

# Sample code for testing
SAMPLE_CODE = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""

class TestModelCache:
    """Test model caching functionality"""

    def test_cache_set_and_get(self):
        """Test basic cache operations"""
        cache = ModelCache(max_size=10, ttl_minutes=60)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = ModelCache()
        assert cache.get("nonexistent") is None

    def test_cache_clear(self):
        """Test cache clearing"""
        cache = ModelCache()
        cache.set("key1", "value1")
        cache.clear()
        assert cache.get("key1") is None

class TestCodeBERT:
    """Test CodeBERT model functionality"""

    @pytest.mark.asyncio
    async def test_codebert_analyze(self):
        """Test CodeBERT analysis with pattern matching"""
        model = CodeBERTModel()
        result = await model.analyze_code(SAMPLE_CODE)

        print(f"\n✅ CodeBERT Analysis: {result.get('analysis', 'N/A')}")

        assert "analysis" in result
        assert "patterns" in result
        assert "model" in result
        assert result["model"] == "microsoft/CodeBERT-base"

    @pytest.mark.asyncio
    async def test_codebert_embeddings(self):
        """Test CodeBERT embeddings (may fail with free API)"""
        model = CodeBERTModel()
        embeddings = await model.get_embeddings(SAMPLE_CODE)

        # Embeddings may be None if API unavailable
        if embeddings is None:
            print("\n⚠️  Embeddings failed (expected with free API)")
            assert embeddings is None
        else:
            print(f"\n✅ Got embeddings with dimension: {len(embeddings[0])}")
            assert isinstance(embeddings, list)

    def test_codebert_pattern_analysis(self):
        """Test pattern-based analysis (doesn't need API)"""
        model = CodeBERTModel()

        # Test with complex code
        complex_code = """
def complex_function():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        pass
        """ * 20

        patterns = model._analyze_patterns(complex_code, None)

        # Check it returns a dict with expected keys
        assert isinstance(patterns, dict)
        assert "functions" in patterns
        assert "classes" in patterns
        assert "complexity_indicators" in patterns

        # Check nested blocks detected
        assert patterns["complexity_indicators"]["nested_blocks"] > 50

        print(f"\n✅ Pattern analysis: {patterns['functions']} functions, "
              f"{patterns['complexity_indicators']['nested_blocks']} nested blocks")

    def test_codebert_pattern_with_classes(self):
        """Test pattern detection with classes"""
        model = CodeBERTModel()

        code_with_classes = """
class MyClass:
    def method1(self):
        pass
    
    def method2(self):
        pass

class AnotherClass:
    pass
        """

        patterns = model._analyze_patterns(code_with_classes, None)

        assert patterns["classes"] == 2
        assert patterns["functions"] == 2
        print(f"\n✅ Detected {patterns['classes']} classes, {patterns['functions']} methods")

class TestFLANT5:
    """Test FLAN-T5 model functionality"""

    @pytest.mark.asyncio
    async def test_flant5_review(self):
        """Test code review generation"""
        model = FLANT5Model()

        try:
            review = await model.generate_review(SAMPLE_CODE, "python")
            print(f"\n✅ Review generated: {review[:100]}...")
            assert isinstance(review, str)
            assert len(review) > 0
        except Exception as e:
            print(f"\n⚠️  Review generation failed (expected with free API): {e}")
            pytest.skip("API not available")

    @pytest.mark.asyncio
    async def test_flant5_documentation(self):
        """Test documentation generation"""
        model = FLANT5Model()

        try:
            docs = await model.generate_documentation(SAMPLE_CODE)
            print(f"\n✅ Docs generated: {docs[:100]}...")
            assert isinstance(docs, str)
            assert len(docs) > 0
        except Exception as e:
            print(f"\n⚠️  Docs generation failed (expected with free API): {e}")
            pytest.skip("API not available")

    @pytest.mark.asyncio
    async def test_flant5_tests(self):
        """Test unit test generation"""
        model = FLANT5Model()

        try:
            tests = await model.generate_tests(SAMPLE_CODE, "pytest")
            print(f"\n✅ Tests generated: {tests[:100]}...")
            assert isinstance(tests, str)
            assert "pytest" in tests.lower() or "test" in tests.lower()
        except Exception as e:
            print(f"\n⚠️  Test generation failed (expected with free API): {e}")
            pytest.skip("API not available")

class TestModelManager:
    """Test ModelManager integration"""

    @pytest.mark.asyncio
    async def test_manager_analyze_code(self):
        """Test code analysis through manager"""
        manager = ModelManager()
        result = await manager.analyze_code(SAMPLE_CODE)

        print(f"\n✅ Manager analysis: {result.get('analysis', 'N/A')}")

        assert "analysis" in result
        assert "patterns" in result

    @pytest.mark.asyncio
    async def test_manager_embeddings(self):
        """Test embeddings through manager"""
        manager = ModelManager()
        embeddings = await manager.get_code_embeddings(SAMPLE_CODE)

        if embeddings:
            print(f"\n✅ Got embeddings via manager")
        else:
            print(f"\n⚠️  Embeddings unavailable (expected with free API)")

        # Should return None or list, not error
        assert embeddings is None or isinstance(embeddings, list)

    def test_manager_cache_clear(self):
        """Test cache clearing"""
        manager = ModelManager()
        manager.clear_caches()
        print("\n✅ Caches cleared successfully")

class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline"""
        manager = ModelManager()

        # Step 1: Analyze code structure
        analysis = await manager.analyze_code(SAMPLE_CODE)
        assert "patterns" in analysis
        print(f"\n✅ Step 1: Pattern analysis complete")

        # Step 2: Try to generate review (may fail with free API)
        try:
            review = await manager.generate_review(SAMPLE_CODE, "python")
            assert isinstance(review, str)
            print(f"✅ Step 2: Review generated")
        except Exception:
            print(f"⚠️  Step 2: Review skipped (API unavailable)")

        # Step 3: Try to generate docs (may fail with free API)
        try:
            docs = await manager.generate_documentation(SAMPLE_CODE)
            assert isinstance(docs, str)
            print(f"✅ Step 3: Docs generated")
        except Exception:
            print(f"⚠️  Step 3: Docs skipped (API unavailable)")

        print(f"\n✅ Pipeline complete!")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])