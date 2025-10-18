"""
Robust Qwen2.5 integration for MCP Developer Workflow

- Primary mode: local transformers (Qwen2.5 embeddings + generation)
- Fallback: Hugging Face Inference API only if local transformers are not available
- Caching for tokenization, model outputs, and embeddings
- Async-safe (uses asyncio.to_thread for blocking calls)
- Exposes QwenModel wrapper and ModelManager for tests and application code
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
    from transformers import AutoTokenizer, AutoModelForCausalLM

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
        """Return cached value if not expired."""
        item = self.cache.get(key)
        if not item:
            return None
        try:
            value, ts = item
        except Exception:
            # defensive fallback — corrupted entry, remove it
            try:
                del self.cache[key]
            except KeyError:
                pass
            return None

        if datetime.now() - ts > self.ttl:
            try:
                del self.cache[key]
            except KeyError:
                pass
            return None
        return value

    def set(self, key: str, value: Any):
        """Insert or update cache entry with timestamp."""
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda kv: kv[1][1])[0]
            try:
                del self.cache[oldest]
            except KeyError:
                pass
        self.cache[key] = (value, datetime.now())

    def clear(self):
        self.cache.clear()


# ---------------------------
# Model singleton + caches
# ---------------------------
@dataclass
class LocalModels:
    qwen_tokenizer: Optional[Any] = None
    qwen_model: Optional[Any] = None


_local = LocalModels()
_token_cache = ModelCache(max_size=500, ttl_minutes=120)
_output_cache = ModelCache(max_size=500, ttl_minutes=120)
_embedding_cache = ModelCache(max_size=1000, ttl_minutes=240)


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
# Initialize local Qwen2.5 (lazy)
# ---------------------------
def _ensure_qwen_loaded():
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Transformers not available for local Qwen2.5.")
    if _local.qwen_model is None:
        logger.info("Loading local Qwen2.5 model...")
        _local.qwen_tokenizer = AutoTokenizer.from_pretrained(
            config.huggingface.model_qwen, cache_dir=config.huggingface.model_cache_dir
        )
        dtype = _torch_dtype()
        device_map = "auto" if torch.cuda.is_available() else None
        _local.qwen_model = AutoModelForCausalLM.from_pretrained(
            config.huggingface.model_qwen,
            torch_dtype=dtype if dtype is not None else None,
            device_map=device_map,
            cache_dir=config.huggingface.model_cache_dir,
        )
        _local.qwen_model.eval()
        logger.info("✅ Local Qwen2.5 ready.")


# ---------------------------
# HF Inference client
# ---------------------------
def _hf_client() -> Optional[Any]:
    if HF_HUB_AVAILABLE and config.huggingface.api_token:
        return InferenceClient(token=config.huggingface.api_token)
    return None


# ---------------------------
# Embedding extraction (async)
# ---------------------------
async def get_qwen_embeddings(text: str) -> Optional[List[float]]:
    if not text:
        return None
    cache_key = f"emb:{hash(text)}"
    cached = _embedding_cache.get(cache_key)
    if cached is not None:
        return cached

    if TRANSFORMERS_AVAILABLE:
        try:
            _ensure_qwen_loaded()
            tok_key = f"tok_qwen:{hash(text)}"
            tokenized = _token_cache.get(tok_key)
            if tokenized is None:
                if _local.qwen_tokenizer is None:
                    raise RuntimeError("Qwen tokenizer not loaded")
                tokenized = _local.qwen_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                _token_cache.set(tok_key, tokenized)
            device = _local.qwen_model.device if _local.qwen_model is not None else _device()
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            def _run():
                with torch.no_grad():
                    outputs = _local.qwen_model(**tokenized)
                    emb_tensor = getattr(outputs, "last_hidden_state", None)
                    if emb_tensor is not None:
                        return emb_tensor.mean(dim=1).squeeze(0).cpu().tolist()
                    elif isinstance(outputs, tuple) and len(outputs) > 0 and hasattr(outputs[0], "mean"):
                        return outputs[0].mean(dim=1).squeeze(0).cpu().tolist()
                    logits = getattr(outputs, "logits", None)
                    if logits is not None:
                        return logits.mean(dim=1).squeeze(0).cpu().tolist()
                    return None

            embeddings = await asyncio.to_thread(_run)
            if embeddings is not None:
                _embedding_cache.set(cache_key, embeddings)
            return embeddings
        except Exception:
            logger.warning("Local Qwen2.5 embedding failed — will try HF fallback.")

    # HF Fallback
    client = _hf_client()
    if client:
        def _call_api() -> Optional[List[float]]:
            resp = client.feature_extraction(text[:512], model=config.huggingface.model_qwen)
            if isinstance(resp, list) and len(resp) and isinstance(resp[0], list):
                return resp[0]
            return None
        embeddings = await asyncio.to_thread(_call_api)  # type: ignore
        if embeddings is not None:
            _embedding_cache.set(cache_key, embeddings)
        return embeddings

    return None

# ---------------------------
# Prompt & generation helpers
# ---------------------------
def _embed_summary_signal(embeddings: Optional[List[float]]) -> str:
    if not embeddings:
        return ""
    import math

    norm = math.sqrt(sum(x * x for x in embeddings) / max(1, len(embeddings)))
    return f"(semantic_score:{norm:.3f})"


def _build_prompt_for(task_type: str, code: str, embed_signal: str) -> str:
    if task_type == "review":
        return (
            f"You are an expert Python code reviewer. {embed_signal}\n\n"
            "Read the code and produce a concise structured review with these sections:\n"
            "1) Summary — one sentence\n"
            "2) Bugs or potential issues — bullet list (or 'None')\n"
            "3) Style / maintainability suggestions — bullet list\n"
            "4) Small example fix or snippet if applicable\n\n"
            f"Code:\n{code}\n\nReview:\n"
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
    return f"Explain the following Python code in clear terms. {embed_signal}\n\nCode:\n{code}\n\nExplanation:\n"


async def _generate_with_local_qwen(prompt: str, max_new_tokens: int = 300) -> str:
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Local Qwen2.5 required but transformers not available.")

    _ensure_qwen_loaded()
    cache_key = f"qwen_out:{hash(prompt)}"
    cached = _output_cache.get(cache_key)
    if cached:
        return cached

    tok_key = f"tok_qwen:{hash(prompt)}"
    tokenized = _token_cache.get(tok_key)
    if tokenized is None:
        if _local.qwen_tokenizer is None:
            raise RuntimeError("Qwen tokenizer not loaded")
        tokenized = _local.qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        _token_cache.set(tok_key, tokenized)

    try:
        device = _local.qwen_model.device if _local.qwen_model is not None else _device()
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
    except Exception:
        pass

    def _run():
        gen_kwargs = dict(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True,
            pad_token_id=(
                _local.qwen_tokenizer.pad_token_id
                if _local.qwen_tokenizer.pad_token_id is not None
                else _local.qwen_tokenizer.eos_token_id
            ),
        )
        outputs = _local.qwen_model.generate(
            **{k: v for k, v in gen_kwargs.items() if v is not None}
        )
        return _local.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        text = await asyncio.to_thread(_run)
    except TypeError as te:
        logger.warning("Generate() params not fully supported: %s — retrying simpler call", te)

        def _run_simple():
            outputs = _local.qwen_model.generate(
                tokenized["input_ids"], max_new_tokens=max_new_tokens
            )
            return _local.qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)

        text = await asyncio.to_thread(_run_simple)

    _output_cache.set(cache_key, text)
    return text


async def _generate_with_hf_api(prompt: str, max_new_tokens: int = 300) -> str:
    client = _hf_client()
    if not client:
        raise RuntimeError("Hugging Face Inference client not configured.")

    def _call():
        return client.text_generation(
            prompt, model=config.huggingface.model_qwen, max_new_tokens=max_new_tokens
        )

    resp = await asyncio.to_thread(_call)
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        return resp.get("generated_text") or str(resp)
    return str(resp)


async def generate_text(
    prompt: str,
    task_type: str = "general",
    embeddings: Optional[List[float]] = None,
    max_new_tokens: int = 300,
) -> str:
    if not prompt:
        return ""
    embed_signal = _embed_summary_signal(embeddings)
    full_prompt = _build_prompt_for(task_type, prompt, embed_signal)

    if TRANSFORMERS_AVAILABLE:
        try:
            return await _generate_with_local_qwen(full_prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.warning("Local Qwen2.5 generation failed: %s", e)

    client = _hf_client()
    if client:
        try:
            return await _generate_with_hf_api(full_prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.error("HF text_generation failed: %s", e)

    logger.error("No generation method available.")
    return ""


# ---------------------------
# QwenModel wrapper (lightweight API for tests & app)
# ---------------------------
class QwenModel:
    """
    Lightweight wrapper around Qwen capabilities.

    - get_embeddings(text): async -> Optional[List[float]]
    - generate_text(prompt, ...): async -> str
    - analyze_code(code): async -> dict (uses embeddings + generation to produce interpretation)
    """

    async def get_embeddings(self, text: str) -> Optional[List[float]]:
        return await get_qwen_embeddings(text)

    async def generate_text(
        self,
        prompt: str,
        task_type: str = "general",
        embeddings: Optional[List[float]] = None,
        max_new_tokens: int = 300,
    ) -> str:
        return await generate_text(
            prompt, task_type=task_type, embeddings=embeddings, max_new_tokens=max_new_tokens
        )

    async def analyze_code(self, code: str) -> Dict[str, Any]:
        # Basic pattern extraction (same as ModelManager uses)
        # Extract counts explicitly as int to help mypy
        functions_count: int = len(re.findall(r"\bdef\s+\w+\s*\(", code))
        classes_count: int = len(re.findall(r"\bclass\s+\w+", code))
        imports_count: int = len(re.findall(r"^\s*(?:from|import)\s+", code, re.MULTILINE))
        comments_count: int = len(re.findall(r"#.*$", code, re.MULTILINE))
        docstrings_count: int = len(re.findall(r'""".*?"""', code, re.DOTALL))

        patterns: Dict[str, Any] = {
            "functions": functions_count,
            "classes": classes_count,
            "imports": imports_count,
            "comments": comments_count,
            "docstrings": docstrings_count,
            "complexity_indicators": {
                "nested_blocks": code.count("    if ")
                + code.count("    for ")
                + code.count("    while "),
                "try_except_blocks": code.count("try:"),
                "loops": code.count("for ") + code.count("while "),
            },
        }

        embeddings = await self.get_embeddings(code)
        analysis_text = None
        if embeddings:
            try:
                # Use Qwen to produce a short interpretation (non-blocking)
                analysis_text = await self.generate_text(
                    code, task_type="general", embeddings=embeddings, max_new_tokens=200
                )
            except Exception as e:
                logger.warning("QwenModel.analysis generation failed: %s", e)

        if not analysis_text:
            parts: List[str] = []
            if functions_count > 0:
                parts.append(f"Contains {functions_count} function(s)")
            if classes_count > 0:
                parts.append(f"Contains {classes_count} class(es)")
            if docstrings_count == 0 and functions_count > 0:
                parts.append("Missing docstrings")
            analysis_text = ". ".join(parts) if parts else "No notable structure detected"

        return {
            "analysis": analysis_text,
            "patterns": patterns,
            "model": config.huggingface.model_qwen,
            "timestamp": datetime.now().isoformat(),
        }


# ---------------------------
# ModelManager
# ---------------------------
class ModelManager:
    def __init__(self):
        logger.info("Initializing ModelManager...")
        # public caches for external control
        self.embedding_cache = _embedding_cache
        self.token_cache = _token_cache
        self.output_cache = _output_cache
        # expose a QwenModel instance for tests and other modules
        self.qwen = QwenModel()
        logger.info("ModelManager ready.")

    async def get_code_embeddings(self, code: str) -> Optional[List[float]]:
        return await self.qwen.get_embeddings(code)

    async def analyze_code(self, code: str) -> Dict[str, Any]:
        return await self.qwen.analyze_code(code)

    async def generate_review(self, code: str, language: str = "python") -> str:
        embeddings = await self.get_code_embeddings(code)
        prompt = code if language.lower() in ("python", "py") else code
        return await self.qwen.generate_text(
            prompt, task_type="review", embeddings=embeddings, max_new_tokens=350
        )

    async def generate_documentation(self, code: str) -> str:
        embeddings = await self.get_code_embeddings(code)
        return await self.qwen.generate_text(
            code, task_type="docs", embeddings=embeddings, max_new_tokens=300
        )

    def _generate_refactoring_suggestions(self, score: float) -> List[str]:
        # Explicit type conversion to ensure mypy understands the numeric type
        score_val = float(score)

        if score_val < 30.0:
            return ["Code complexity is acceptable"]
        elif score_val < 60.0:
            return ["Consider breaking down large functions", "Review nesting levels"]
        else:
            return ["High complexity detected", "Refactor into smaller modules",
                    "Extract complex logic into separate functions"]

    async def generate_tests(self, code: str) -> str:
        embeddings = await self.get_code_embeddings(code)
        return await self.qwen.generate_text(
            code, task_type="tests", embeddings=embeddings, max_new_tokens=400
        )

    def clear_caches(self):
        self.embedding_cache.clear()
        self.token_cache.clear()
        self.output_cache.clear()
        try:
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("All model caches cleared.")


# singleton instance
model_manager = ModelManager()