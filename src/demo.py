# file: demo.py
"""
Demo for MCP Tools: CodeBERT + FLAN-T5 integration
Tests models, tools, and resources end-to-end
"""

import asyncio
from src.models import ModelManager

# Optional: placeholder tools and resources imports
# from src.tools import ToolManager
# from src.resources import ResourceManager

async def main():
    print("🚀 Starting MCP demo with CodeBERT + FLAN-T5...")

    # -----------------------------
    # 1. Initialize ModelManager
    # -----------------------------
    model_manager = ModelManager()

    # Sample code snippet for testing
    sample_code = """
def reverse_string(s):
    \"\"\"Reverses the input string\"\"\"
    return s[::-1]

class StringUtils:
    pass
"""

    # -----------------------------
    # 2. Test CodeBERT embeddings
    # -----------------------------
    embeddings = await model_manager.get_code_embeddings(sample_code)
    if embeddings is not None:
        print(f"\n✅ CodeBERT embeddings generated (dim={len(embeddings)})")
    else:
        print("\n⚠️ CodeBERT embeddings failed")

    if embeddings:
        print(f"\n✅ CodeBERT embeddings generated (dim={len(embeddings)})")
    else:
        print("\n⚠️ CodeBERT embeddings failed")

    # -----------------------------
    # 3. Test FLAN-T5 code review
    # -----------------------------
    review = await model_manager.generate_review(sample_code, language="python")
    print("\n💡 FLAN-T5 Code Review:")
    print(review)

    # -----------------------------
    # 4. Test FLAN-T5 documentation generation
    # -----------------------------
    documentation = await model_manager.generate_documentation(sample_code)
    print("\n📄 FLAN-T5 Documentation:")
    print(documentation)

    # -----------------------------
    # 5. Test FLAN-T5 unit test generation
    # -----------------------------
    tests = await model_manager.generate_tests(sample_code, framework="pytest")
    print("\n🧪 FLAN-T5 Generated Tests:")
    print(tests)

    # -----------------------------
    # 6. Clear caches and re-test
    # -----------------------------
    model_manager.clear_caches()
    print("\n🗑️ Model caches cleared.")

    # Re-run embedding to confirm caching
    embeddings2 = await model_manager.get_code_embeddings(sample_code)
    print(f"\n✅ CodeBERT embeddings re-generated (dim={len(embeddings2)})")

if __name__ == "__main__":
    asyncio.run(main())
