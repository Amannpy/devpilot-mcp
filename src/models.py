# file: src/models.py
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1. GLOBAL MODEL CACHE (loaded once)
# ---------------------------------------------------------------------
logger.info("🔄 Loading CodeBERT + FLAN-T5 models...")

# --- CodeBERT (for embeddings)
BERT_MODEL_NAME = "microsoft/codebert-base"
tokenizer_bert = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
model_bert = AutoModel.from_pretrained(BERT_MODEL_NAME)
model_bert.eval()
logger.info("✅ CodeBERT loaded successfully (local).")

# --- FLAN-T5 (for reasoning)
FLAN_MODEL_NAME = "google-t5/t5-base"
tokenizer_flan = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME)
model_flan = AutoModelForSeq2SeqLM.from_pretrained(
    FLAN_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model_flan.eval()
logger.info("✅ FLAN-T5 loaded successfully (local).")

# Caches for tokenized inputs
bert_cache = {}
flan_cache = {}

# ---------------------------------------------------------------------
# 2. CODEBERT EMBEDDING FUNCTION
# ---------------------------------------------------------------------
def get_codebert_embeddings(text: str):
    """Generate mean-pooled embeddings for given code or text using CodeBERT."""
    if not text:
        return None

    if text in bert_cache:
        inputs = bert_cache[text]
    else:
        inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, max_length=512)
        bert_cache[text] = inputs

    with torch.no_grad():
        outputs = model_bert(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings


# ---------------------------------------------------------------------
# 3. FLAN-T5 GENERATION WRAPPER
# ---------------------------------------------------------------------
def generate_flan_response(prompt: str, task_type: str = "general", embeddings=None):
    """Generate structured text response from FLAN-T5 for a given task type."""
    if not prompt:
        return "No input provided."

    # Optional: summarize embedding for context signal
    embed_context = ""
    if embeddings is not None:
        # Normalize and summarize to low-dim signal
        avg_val = torch.mean(F.normalize(embeddings, dim=1)).item()
        embed_context = f"(code complexity: {avg_val:.3f})"

    # Task-specific prompting
    if task_type == "review":
        t5_input = f"Perform a professional code review. Identify bugs, style issues, and improvements {embed_context}:\n{prompt}"
    elif task_type == "docs":
        t5_input = f"Generate clear and concise documentation or docstrings for this code {embed_context}:\n{prompt}"
    elif task_type == "tests":
        t5_input = f"Write meaningful unit tests in Python for this code {embed_context}:\n{prompt}"
    else:
        t5_input = f"Explain the following code in simple terms {embed_context}:\n{prompt}"

    if t5_input in flan_cache:
        inputs = flan_cache[t5_input]
    else:
        inputs = tokenizer_flan(t5_input, return_tensors="pt", truncation=True, max_length=512).to(model_flan.device)
        flan_cache[t5_input] = inputs

    with torch.no_grad():
        outputs = model_flan.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9
        )

    decoded = tokenizer_flan.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


# ---------------------------------------------------------------------
# 4. MASTER INFERENCE FUNCTIONS (used by demo.py)
# ---------------------------------------------------------------------
def analyze_code(code: str):
    """Performs full analysis using both CodeBERT and FLAN-T5."""
    result = {
        "embeddings": None,
        "review": "Generation failed",
        "docs": "Generation failed",
        "tests": "Generation failed",
    }

    try:
        embeddings = get_codebert_embeddings(code)
        result["embeddings"] = embeddings
        logger.info("✅ CodeBERT embeddings generated (dim=%d)", embeddings.shape[1])
    except Exception as e:
        logger.error("CodeBERT embedding generation failed: %s", e)
        return result

    # --- Run tasks ---
    try:
        result["review"] = generate_flan_response(code, task_type="review", embeddings=embeddings)
    except Exception as e:
        logger.error("FLAN-T5 review failed: %s", e)

    try:
        result["docs"] = generate_flan_response(code, task_type="docs", embeddings=embeddings)
    except Exception as e:
        logger.error("FLAN-T5 docs failed: %s", e)

    try:
        result["tests"] = generate_flan_response(code, task_type="tests", embeddings=embeddings)
    except Exception as e:
        logger.error("FLAN-T5 tests failed: %s", e)

    return result


# ---------------------------------------------------------------------
# 5. CACHE CLEAR FUNCTION
# ---------------------------------------------------------------------
def clear_model_caches():
    """Clear cached tokenizations and free GPU memory."""
    bert_cache.clear()
    flan_cache.clear()
    torch.cuda.empty_cache()
    logger.info("🗑️ Model caches cleared successfully.")
