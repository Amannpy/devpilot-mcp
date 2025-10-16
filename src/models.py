# models.py
"""
AI Model integrations for code analysis
Handles Hugging Face model inference, local caching, and async calls
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import re

from src.config import config

logger = logging.getLogger(__name__)

# -----------------------------
# Transformers availability
# -----------------------------
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available - embeddings will be disabled")

# -----------------------------
# Model cache
# -----------------------------
class ModelCache:
    """In-memory cache for model responses"""
    def __init__(self, max_size: int = 100, ttl_minutes: int = 60):
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                logger.debug(f"Cache hit for key: {key[:50]}")
                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (value, datetime.now())
        logger.debug(f"Cached value for key: {key[:50]}")

    def clear(self):
        self.cache.clear()
        logger.info("Cache cleared")

# -----------------------------
# CodeBERT embeddings
# -----------------------------
class CodeBERTModel:
    """Generate embeddings for code using CodeBERT (local transformer)"""
    MODEL_NAME = config.huggingface.model_code_bert
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModel] = None
    cache = ModelCache(ttl_minutes=60)

    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            if not self.__class__.tokenizer:
                self.__class__.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                self.__class__.model = AutoModel.from_pretrained(self.MODEL_NAME)
                self.__class__.model.eval()
                logger.info("✅ CodeBERT loaded (local).")
        else:
            raise RuntimeError("Transformers library is required for CodeBERT embeddings.")

    async def get_embeddings(self, code: str) -> Optional[List[float]]:
        """Generate embeddings for the given code"""
        cache_key = hash(code)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            inputs = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()[0]
            self.cache.set(cache_key, embeddings)
            logger.info(f"✅ Local embeddings generated, dim={len(embeddings)}")
            return embeddings
        except Exception as e:
            logger.error(f"CodeBERT embedding generation failed: {e}")
            return None

    async def analyze_patterns(self, code: str, embeddings: Optional[List[float]] = None) -> Dict[str, Any]:
        """Basic pattern-based code analysis"""
        patterns = {
            "functions": len(re.findall(r'\bdef\s+\w+\s*\(', code)),
            "classes": len(re.findall(r'\bclass\s+\w+', code)),
            "imports": len(re.findall(r'^\s*(?:from|import)\s+', code, re.MULTILINE)),
            "comments": len(re.findall(r'#.*$', code, re.MULTILINE)),
            "docstrings": len(re.findall(r'""".*?"""', code, re.DOTALL)),
            "complexity": {
                "nested_blocks": code.count('    if ') + code.count('    for ') + code.count('    while '),
                "loops": code.count('for ') + code.count('while '),
            }
        }
        if embeddings:
            patterns["has_semantic_info"] = True
        return patterns

# -----------------------------
# FLAN-T5 text generation
# -----------------------------
class FLANT5Model:
    """FLAN-T5 text generation (local)"""
    MODEL_NAME = config.huggingface.model_flan_t5
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForSeq2SeqLM] = None
    cache = ModelCache(ttl_minutes=60)

    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            if not self.__class__.tokenizer:
                self.__class__.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                self.__class__.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.MODEL_NAME,
                    dtype=torch.float16,
                    device_map="auto"
                )
                self.__class__.model.eval()
                logger.info("✅ FLAN-T5 loaded (local).")
        else:
            raise RuntimeError("Transformers library is required for FLAN-T5 generation.")

    async def generate_text(self, prompt: str, max_new_tokens: int = 500) -> str:
        """Generate text from prompt"""
        cache_key = hash(prompt)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.cache.set(cache_key, text)
            return text
        except Exception as e:
            logger.error(f"FLAN-T5 generation failed: {e}")
            return "Generation failed"

# -----------------------------
# Model Manager
# -----------------------------
class ModelManager:
    """Unified interface for CodeBERT and FLAN-T5"""
    def __init__(self):
        self.codebert = CodeBERTModel()
        self.flant5 = FLANT5Model()
        logger.info("✅ ModelManager initialized")

    async def get_code_embeddings(self, code: str) -> Optional[List[float]]:
        return await self.codebert.get_embeddings(code)

    async def analyze_code(self, code: str) -> Dict[str, Any]:
        embeddings = await self.codebert.get_embeddings(code)
        patterns = await self.codebert.analyze_patterns(code, embeddings)
        return {
            "embeddings_available": embeddings is not None,
            "embedding_dimension": len(embeddings) if embeddings else 0,
            "patterns": patterns
        }

    async def generate_review(self, code: str, language: str = "python") -> str:
        prompt = f"Review this {language} code and provide suggestions:\n{code[:1500]}"
        return await self.flant5.generate_text(prompt, max_new_tokens=500)

    async def generate_documentation(self, code: str) -> str:
        prompt = f"Generate detailed documentation for this code:\n{code[:1500]}"
        return await self.flant5.generate_text(prompt, max_new_tokens=800)

    async def generate_tests(self, code: str, framework: str = "pytest") -> str:
        prompt = f"Generate {framework} unit tests for this code:\n{code[:1000]}"
        return await self.flant5.generate_text(prompt, max_new_tokens=1000)

    def clear_caches(self):
        self.codebert.cache.clear()
        self.flant5.cache.clear()
        logger.info("✅ All caches cleared")

# -----------------------------
# Global instance
# -----------------------------
if __name__ != "__main__":
    model_manager = ModelManager()
