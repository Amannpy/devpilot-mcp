# file: query_bert_flan_cached_v3.py

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# -----------------------------
# 1. Load models globally (cached)
# -----------------------------
print("Loading models, please wait...")

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# CodeBERT for embeddings
BERT_MODEL_NAME = "microsoft/codebert-base"
tokenizer_bert = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
model_bert = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
model_bert.eval()  # eval mode
print("✅ CodeBERT loaded on", device)

# FLAN-T5 for text generation
FLAN_MODEL_NAME = "google-t5/t5-base"
tokenizer_flan = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME)
model_flan = AutoModelForSeq2SeqLM.from_pretrained(
    FLAN_MODEL_NAME,
    device_map="auto" if device=="cuda" else None,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
model_flan.eval()
print("✅ FLAN-T5 loaded")

# -----------------------------
# 2. Tokenization cache
# -----------------------------
bert_token_cache = {}
flan_token_cache = {}

# -----------------------------
# 3. Function to get CodeBERT embeddings
# -----------------------------
def get_bert_embeddings(text: str) -> torch.Tensor:
    """Generate embeddings for the input text using CodeBERT (cached)."""
    try:
        if text in bert_token_cache:
            inputs = bert_token_cache[text]
        else:
            inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            bert_token_cache[text] = inputs

        with torch.no_grad():
            outputs = model_bert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        return embeddings
    except Exception as e:
        raise RuntimeError(f"CodeBERT embedding generation failed: {e}")

# -----------------------------
# 4. Function to generate FLAN-T5 response
# -----------------------------
def generate_flan_response(prompt: str, max_tokens: int = 150) -> str:
    """Generate a response using FLAN-T5 (cached tokenization)."""
    try:
        t5_input = f"summarize: {prompt}"

        if t5_input in flan_token_cache:
            inputs = flan_token_cache[t5_input]
        else:
            inputs = tokenizer_flan(t5_input, return_tensors="pt").to(model_flan.device)
            flan_token_cache[t5_input] = inputs

        outputs = model_flan.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer_flan.pad_token_id
        )
        decoded = tokenizer_flan.decode(outputs[0], skip_special_tokens=True)
        return decoded
    except Exception as e:
        raise RuntimeError(f"FLAN-T5 generation failed: {e}")

# -----------------------------
# 5. Main interactive loop
# -----------------------------
def main():
    print("\nWelcome to CodeBERT + FLAN-T5 Demo (cached models + tokenization)!")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nEnter your query/code snippet: ").strip()
            if not user_input:
                print("⚠️  Empty input, please enter a query or code snippet.")
                continue
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            # Step 1: Generate embeddings
            embeddings = get_bert_embeddings(user_input)
            print(f"\n✅ CodeBERT embeddings generated (dim={embeddings.shape[1]})")

            # Step 2: Generate FLAN-T5 response
            response = generate_flan_response(user_input)
            print(f"\n💡 FLAN-T5 Response:\n{response}")

        except Exception as e:
            print(f"\n⚠️  Error: {e}")

if __name__ == "__main__":
    main()
