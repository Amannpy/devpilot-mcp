# file: models.py
"""
AI Model integrations for code analysis
Handles Hugging Face model inference and caching
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging
import re
from huggingface_hub import InferenceClient
from src.config import config

logger = logging.getLogger(__name__)

# Try to import transformers for local model usage
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available - embeddings will be disabled")


class ModelCache:
    """Simple in-memory cache for model responses"""

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
            else:
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


class HuggingFaceModel:
    """Base class for Hugging Face model integration"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = InferenceClient(token=config.huggingface.api_token)
        self.cache = ModelCache(
            max_size=config.server.cache_size,
            ttl_minutes=60
        )

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using the model"""
        cache_key = f"{self.model_name}:{hash(prompt)}"

        if config.huggingface.use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

        try:
            def _call_api():
                return self.client.text_generation(
                    prompt,
                    model=self.model_name,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.95
                )

            response = await asyncio.to_thread(_call_api)

            if config.huggingface.use_cache and response:
                self.cache.set(cache_key, response)

            return response if response else ""
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return ""


class CodeBERTModel(HuggingFaceModel):
    """CodeBERT model for code understanding"""

    def __init__(self):
        super().__init__(config.huggingface.model_code_bert)

    async def get_embeddings(self, code: str) -> Optional[List[float]]:
        """Get embeddings for a given code snippet"""
        try:
            # Try Hugging Face Inference API first
            response = self.client.feature_extraction(code[:512], model=self.model_name)
            if isinstance(response, list) and isinstance(response[0], list):
                return response[0]
        except Exception as e:
            logger.warning(f"CodeBERT embedding extraction failed (API): {e}")

        # Fallback: Local transformer
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("⚙️ Falling back to local CodeBERT embeddings...")
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                model.eval()
                inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings_tensor = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_size]
                embeddings = embeddings_tensor.squeeze(0).tolist()          # flatten to list
                logger.info(f"✅ Local embeddings generated, dimension = {len(embeddings)}")
                return embeddings
            except Exception as local_e:
                logger.error(f"Local embedding generation failed: {local_e}")

        return None

    async def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure and patterns using embeddings"""
        logger.info("Starting CodeBERT code analysis")
        embeddings = await self.get_embeddings(code)
        patterns = self._analyze_patterns(code, embeddings)
        result = {
            "has_embeddings": embeddings is not None,
            "patterns": patterns,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat()
        }
        if embeddings:
            result["embedding_dimension"] = len(embeddings)
            result["analysis"] = self._interpret_embeddings(code, patterns)
        else:
            result["analysis"] = self._fallback_analysis(patterns)
        return result

    def _analyze_patterns(self, code: str, embeddings: Optional[List[float]]) -> Dict[str, Any]:
        patterns = {
            "functions": len(re.findall(r'\bdef\s+\w+\s*\(', code)),
            "classes": len(re.findall(r'\bclass\s+\w+', code)),
            "imports": len(re.findall(r'^\s*(?:from|import)\s+', code, re.MULTILINE)),
            "comments": len(re.findall(r'#.*$', code, re.MULTILINE)),
            "docstrings": len(re.findall(r'""".*?"""', code, re.DOTALL)),
            "complexity_indicators": {
                "nested_blocks": code.count('    if ') + code.count('    for ') + code.count('    while '),
                "try_except_blocks": code.count('try:'),
                "loops": code.count('for ') + code.count('while '),
            }
        }
        if embeddings:
            patterns["has_semantic_info"] = True
        return patterns

    def _interpret_embeddings(self, code: str, patterns: Dict[str, Any]) -> str:
        analysis_parts = []
        if patterns["functions"] > 0:
            analysis_parts.append(f"Contains {patterns['functions']} function(s)")
        if patterns["classes"] > 0:
            analysis_parts.append(f"Contains {patterns['classes']} class(es)")
        complexity = patterns["complexity_indicators"]
        if complexity["nested_blocks"] > 5:
            analysis_parts.append("High nesting complexity detected")
        if patterns["docstrings"] == 0 and patterns["functions"] > 0:
            analysis_parts.append("Missing docstrings")
        return ". ".join(analysis_parts) if analysis_parts else "Code structure analyzed"

    def _fallback_analysis(self, patterns: Dict[str, Any]) -> str:
        return f"Pattern-based analysis: {patterns['functions']} functions, {patterns['classes']} classes"


class FLANT5Model(HuggingFaceModel):
    """FLAN-T5 model for natural language generation"""

    def __init__(self):
        super().__init__(config.huggingface.model_flan_t5)
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(config.huggingface.model_flan_t5)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                config.huggingface.model_flan_t5,
                dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
            self.token_cache = {}

    async def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using FLAN-T5 (local fallback)"""
        if TRANSFORMERS_AVAILABLE:
            inputs = self.token_cache.get(prompt)
            if not inputs:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                self.token_cache[prompt] = inputs
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Fallback to HF API
        return await self.generate(prompt, max_tokens=max_tokens)


class ModelManager:
    """Manages all AI models"""

    def __init__(self):
        self.codebert = CodeBERTModel()
        self.flant5 = FLANT5Model()
        logger.info("Model manager initialized")

    async def analyze_code(self, code: str) -> Dict[str, Any]:
        return await self.codebert.analyze_code(code)

    async def get_code_embeddings(self, code: str) -> Optional[List[float]]:
        return await self.codebert.get_embeddings(code)

    async def generate_review(self, code: str, language: str) -> str:
        prompt = f"Review this {language} code and provide feedback:\n{code[:1500]}"
        return await self.flant5.generate_text(prompt, max_tokens=500)

    async def generate_documentation(self, code: str) -> str:
        prompt = f"Generate documentation for this code:\n{code[:1500]}"
        return await self.flant5.generate_text(prompt, max_tokens=800)

    async def generate_tests(self, code: str, framework: str) -> str:
        prompt = f"Generate {framework} unit tests for this code:\n{code[:1000]}"
        return await self.flant5.generate_text(prompt, max_tokens=1000)

    def clear_caches(self):
        self.codebert.cache.clear()
        if hasattr(self.flant5, "token_cache"):
            self.flant5.token_cache.clear()
        self.flant5.cache.clear()
        logger.info("All model caches cleared")


# Global model manager instance
if __name__ != "__main__":
    model_manager = ModelManager()
