# src/models.py
"""
Robust model integration for MCP Developer Workflow

- Primary mode: local transformers (CodeBERT embeddings + google/flan-t5-base generation)
- Fallback: Hugging Face Inference API only if local transformers are not available
- Caching for tokenization, model outputs, and embeddings
- Async-safe (uses asyncio.to_thread for blocking calls)
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# local imports and optional HF client
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSeq2SeqLM
    )
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers or torch not available — will try HF Inference API fallback.")

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except Exception:
    HF_HUB_AVAILABLE = False

from src.config import config

# ---------------------------
# Simple in-memory cache
# ---------------------------
class ModelCache:
    def __init__(self, max_size: int = 200, ttl_minutes: int = 60):
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)

    def get(self, key: str) -> Optional[Any]:
        item = self.cache.get(key)
        if not item:
            return None
        value, ts = item
        if datetime.now() - ts > self.ttl:
            del self.cache[key]
            return None
        return value

    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            # evict oldest
            oldest = min(self.cache.items(), key=lambda kv: kv[1][1])[0]
            del self.cache[oldest]
        self.cache[key] = (value, datetime.now())

    def clear(self):
        self.cache.clear()

# ---------------------------
# Model singletons + caches
# ---------------------------
@dataclass
class LocalModels:
    # will be populated on first use
    codebert_tokenizer: Optional[Any] = None
    codebert_model: Optional[Any] = None
    flan_tokenizer: Optional[Any] = None
    flan_model: Optional[Any] = None

_local = LocalModels()
_token_cache = ModelCache(max_size=500, ttl_minutes=120)   # tokenization cache (text->inputs)
_output_cache = ModelCache(max_size=500, ttl_minutes=120)  # generation outputs
_embedding_cache = ModelCache(max_size=1000, ttl_minutes=240)  # embeddings (text->vector)

# ---------------------------
# Helpers: device & dtype
# ---------------------------
def _device():
    try:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    except Exception:
        return "cpu"

def _torch_dtype():
    try:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    except Exception:
        return None

# ---------------------------
# Initialize local models (lazy)
# ---------------------------
def _ensure_codebert_loaded():
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers not available for local CodeBERT.")
    if _local.codebert_model is None:
        logger.info("Loading local CodeBERT model...")
        _local.codebert_tokenizer = AutoTokenizer.from_pretrained(config.huggingface.model_code_bert)
        _local.codebert_model = AutoModel.from_pretrained(config.huggingface.model_code_bert)
        _local.codebert_model.eval()
        # move to device if GPU available
        try:
            _local.codebert_model.to(_device())
        except Exception:
            pass
        logger.info("✅ Local CodeBERT ready.")

def _ensure_flan_loaded():
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers not available for local FLAN-T5.")
    if _local.flan_model is None:
        logger.info("Loading local FLAN-T5 model...")
        _local.flan_tokenizer = AutoTokenizer.from_pretrained(config.huggingface.model_flan_t5)
        dtype = _torch_dtype()
        # device_map only when cuda exists; otherwise load to CPU
        device_map = "auto" if torch.cuda.is_available() else None
        _local.flan_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.huggingface.model_flan_t5,
            torch_dtype=dtype if dtype is not None else None,
            device_map=device_map
        )
        _local.flan_model.eval()
        logger.info("✅ Local FLAN-T5 ready.")

# ---------------------------
# Fallback HF Inference client (if needed)
# ---------------------------
def _hf_client() -> Optional[Any]:
    if HF_HUB_AVAILABLE and config.huggingface.api_token:
        return InferenceClient(token=config.huggingface.api_token)
    return None

# ---------------------------
# Embedding extraction (async)
# ---------------------------
async def get_codebert_embeddings(code: str) -> Optional[List[float]]:
    """Return a flat list[float] embedding for code/text."""
    if not code:
        return None

    cache_key = f"emb:{hash(code)}"
    cached = _embedding_cache.get(cache_key)
    if cached:
        return cached

    # Prefer local
    if TRANSFORMERS_AVAILABLE:
        try:
            _ensure_codebert_loaded()
            # tokenization caching
            tok_key = f"tok_bert:{hash(code)}"
            tokenized = _token_cache.get(tok_key)
            if tokenized is None:
                tokenized = _local.codebert_tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
                _token_cache.set(tok_key, tokenized)

            # ensure tensors on same device as model
            try:
                device = _device()
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
            except Exception:
                pass

            # run model in thread to avoid blocking event loop
            def _run():
                with torch.no_grad():
                    outputs = _local.codebert_model(**tokenized)
                    emb_tensor = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    return emb_tensor.cpu().tolist()
            embeddings = await asyncio.to_thread(_run)
            _embedding_cache.set(cache_key, embeddings)
            return embeddings
        except Exception as e:
            logger.warning("Local CodeBERT failed (%s). Will try HF Inference fallback if available.", e)

    # HF Inference API fallback (if configured)
    client = _hf_client()
    if client:
        try:
            def _call_api():
                resp = client.feature_extraction(code[:512], model=config.huggingface.model_code_bert)
                # expecting [[...]] -> return first list
                if isinstance(resp, list) and len(resp) and isinstance(resp[0], list):
                    return resp[0]
                return None
            embeddings = await asyncio.to_thread(_call_api)
            if embeddings:
                _embedding_cache.set(cache_key, embeddings)
            return embeddings
        except Exception as e:
            logger.error("HF feature_extraction failed: %s", e)

    logger.error("No method available to compute embeddings (local transformers or HF token missing).")
    return None

# ---------------------------
# Prompt templates and generation helper
# ---------------------------
def _embed_summary_signal(embeddings: Optional[List[float]]) -> str:
    if not embeddings:
        return ""
    try:
        # reduce to a small scalar to include in prompt (semantic signal)
        import math
        # compute L2 norm / dim
        norm = math.sqrt(sum(x * x for x in embeddings) / max(1, len(embeddings)))
        return f"(semantic_score:{norm:.3f})"
    except Exception:
        return ""

def _build_prompt_for(task_type: str, code: str, embed_signal: str) -> str:
    if task_type == "review":
        return (
            f"You are an expert Python code reviewer. {embed_signal}\n\n"
            "Read the code and produce a concise structured review with these sections:\n"
            "1) Summary — one sentence\n"
            "2) Bugs or potential issues — bullet list (or 'None')\n"
            "3) Style / maintainability suggestions — bullet list\n"
            "4) Small example fix or snippet if applicable\n\n"
            "Code:\n"
            f"{code}\n\nReview:\n"
        )
    if task_type == "docs":
        return (
            f"You are an expert technical writer for Python code. {embed_signal}\n\n"
            "Produce a docstring and a short Markdown usage example for the following code.\n\n"
            f"Code:\n{code}\n\nDocumentation:\n"
        )
    if task_type == "tests":
        return (
            f"You are an expert in writing pytest unit tests. {embed_signal}\n\n"
            "Generate pytest unit tests covering typical and edge cases for the following code.\n\n"
            f"Code:\n{code}\n\nTests:\n"
        )
    # default / explain
    return (
        f"Explain the following Python code in clear terms. {embed_signal}\n\n"
        f"Code:\n{code}\n\nExplanation:\n"
    )

async def _generate_with_local_flan(prompt: str, max_new_tokens: int = 300) -> str:
    """Use local FLAN-T5 generate with safe parameters (async-friendly)."""
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Local FLAN-T5 required but transformers not available.")

    _ensure_flan_loaded()

    cache_key = f"flan_out:{hash(prompt)}"
    cached = _output_cache.get(cache_key)
    if cached:
        return cached

    # tokenization cache
    tok_key = f"tok_flan:{hash(prompt)}"
    tokenized = _token_cache.get(tok_key)
    if tokenized is None:
        tokenized = _local.flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        _token_cache.set(tok_key, tokenized)

    # move to model device
    try:
        device = _local.flan_model.device if hasattr(_local.flan_model, "device") else _device()
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
    except Exception:
        pass

    def _run():
        # attempt to use beam search + repetition controls
        gen_kwargs = dict(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True,
            pad_token_id=_local.flan_tokenizer.pad_token_id if _local.flan_tokenizer.pad_token_id is not None else _local.flan_tokenizer.eos_token_id,
        )
        # Some HF versions expect positional args; pass via expand call
        outputs = _local.flan_model.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})
        return _local.flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        text = await asyncio.to_thread(_run)
    except TypeError as te:
        # fallback simple generate without extra kwargs
        logger.warning("Generate() params not fully supported: %s — retrying simpler call", te)
        def _run_simple():
            outputs = _local.flan_model.generate(tokenized["input_ids"], max_new_tokens=max_new_tokens)
            return _local.flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = await asyncio.to_thread(_run_simple)

    _output_cache.set(cache_key, text)
    return text

async def _generate_with_hf_api(prompt: str, max_new_tokens: int = 300) -> str:
    client = _hf_client()
    if not client:
        raise RuntimeError("Hugging Face Inference client not configured.")
    # HF InferenceClient text_generation returns model outputs (dict/list); we convert to string
    def _call():
        return client.text_generation(prompt, model=config.huggingface.model_flan_t5, max_new_tokens=max_new_tokens)
    resp = await asyncio.to_thread(_call)
    # resp can be string or dict; try to extract generated text robustly
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        # some API variants return {'generated_text': '...'}
        return resp.get("generated_text") or str(resp)
    # fallback
    return str(resp)

async def generate_text(prompt: str, task_type: str = "general", embeddings: Optional[List[float]] = None, max_new_tokens: int = 300) -> str:
    """
    High-level generator: tries local FLAN-T5 first, then HF API fallback.
    """
    if not prompt:
        return ""

    embed_signal = _embed_summary_signal(embeddings)
    full_prompt = _build_prompt_for(task_type, prompt, embed_signal)

    # Prefer local generation
    if TRANSFORMERS_AVAILABLE:
        try:
            return await _generate_with_local_flan(full_prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.warning("Local FLAN-T5 generation failed: %s", e)

    # Try HF API fallback (only if token provided)
    client = _hf_client()
    if client:
        try:
            return await _generate_with_hf_api(full_prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.error("HF text_generation failed: %s", e)

    logger.error("No generation method available.")
    return ""

# ---------------------------
# ModelManager (API expected by server)
# ---------------------------
class ModelManager:
    def __init__(self):
        logger.info("Initializing ModelManager...")
        # no heavy-loading here; lazy when first called
        # keep caches accessible
        self.codebert_cache = _embedding_cache
        self.token_cache = _token_cache
        self.output_cache = _output_cache
        logger.info("ModelManager ready.")

    async def get_code_embeddings(self, code: str) -> Optional[List[float]]:
        return await get_codebert_embeddings(code)

    async def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Returns pattern analysis + (if available) short interpretation.
        """
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

        embeddings = await self.get_code_embeddings(code)
        analysis_text = None
        if embeddings:
            # produce a short interpretation using generate_text (non-blocking)
            try:
                analysis_text = await generate_text(code, task_type="general", embeddings=embeddings, max_new_tokens=200)
            except Exception as e:
                logger.warning("Interpretation generation failed: %s", e)

        if not analysis_text:
            # fallback short textual analysis
            parts = []
            if patterns["functions"] > 0:
                parts.append(f"Contains {patterns['functions']} function(s)")
            if patterns["classes"] > 0:
                parts.append(f"Contains {patterns['classes']} class(es)")
            if patterns["docstrings"] == 0 and patterns["functions"] > 0:
                parts.append("Missing docstrings")
            analysis_text = ". ".join(parts) if parts else "No notable structure detected"

        return {
            "analysis": analysis_text,
            "patterns": patterns,
            "model": config.huggingface.model_code_bert,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_review(self, code: str, language: str = "python") -> str:
        embeddings = await self.get_code_embeddings(code)
        prompt = code if language.lower() in ("python", "py") else code
        return await generate_text(prompt, task_type="review", embeddings=embeddings, max_new_tokens=350)

    async def generate_documentation(self, code: str) -> str:
        embeddings = await self.get_code_embeddings(code)
        return await generate_text(code, task_type="docs", embeddings=embeddings, max_new_tokens=300)

    async def generate_tests(self, code: str, framework: str = "pytest") -> str:
        embeddings = await self.get_code_embeddings(code)
        # The 'tests' template writes pytest tests by default
        tests_text = await generate_text(code, task_type="tests", embeddings=embeddings, max_new_tokens=400)
        return tests_text

    def clear_caches(self):
        self.codebert_cache.clear()
        self.token_cache.clear()
        self.output_cache.clear()
        # clear CUDA cache if available
        try:
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("All model caches cleared.")

# singleton instance for quick imports
model_manager = ModelManager()
