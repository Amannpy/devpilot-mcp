# file: demo.py
"""
Demo script for testing the MCP AI models (CodeBERT + FLAN-T5)
"""

import asyncio
from src.models import ModelManager

async def main():
    print("🚀 Starting MCP demo with CodeBERT + FLAN-T5...")

    manager = ModelManager()

    # --- Sample input code ---
    sample_code = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""

    # ----------------------------
    # 1️⃣ Code Analysis
    # ----------------------------
    try:
        analysis = await manager.analyze_code(sample_code)
        print("\n🔍 Code Analysis:")
        print(analysis.get("analysis", "No analysis available"))
    except Exception as e:
        print(f"\n❌ Code analysis failed: {e}")

    # ----------------------------
    # 2️⃣ Code Embeddings
    # ----------------------------
    try:
        embeddings = await manager.get_code_embeddings(sample_code)
        if embeddings is not None:
            print(f"\n✅ CodeBERT embeddings generated (dim={len(embeddings)})")
        else:
            print("\n⚠️  CodeBERT embeddings unavailable")
    except Exception as e:
        print(f"\n❌ CodeBERT embedding generation failed: {e}")
        embeddings = None

    # ----------------------------
    # 3️⃣ Code Review
    # ----------------------------
    try:
        review = await manager.generate_review(sample_code, "python")
        if review.strip():
            print("\n💡 FLAN-T5 Code Review:")
            print(review.strip())
        else:
            print("\n⚠️  No review generated")
    except Exception as e:
        print(f"\n❌ FLAN-T5 review generation failed: {e}")

    # ----------------------------
    # 4️⃣ Documentation
    # ----------------------------
    try:
        docs = await manager.generate_documentation(sample_code)
        if docs.strip():
            print("\n📄 FLAN-T5 Documentation:")
            print(docs.strip())
        else:
            print("\n⚠️  No documentation generated")
    except Exception as e:
        print(f"\n❌ FLAN-T5 documentation generation failed: {e}")

    # ----------------------------
    # 5️⃣ Unit Test Generation
    # ----------------------------
    try:
        tests = await manager.generate_tests(sample_code, "pytest")
        if tests.strip():
            print("\n🧪 FLAN-T5 Generated Tests:")
            print(tests.strip())
        else:
            print("\n⚠️  No tests generated")
    except Exception as e:
        print(f"\n❌ FLAN-T5 test generation failed: {e}")

    # ----------------------------
    # 6️⃣ Clear caches
    # ----------------------------
    try:
        manager.clear_caches()
        print("\n🗑️  Model caches cleared successfully.")
    except Exception as e:
        print(f"\n⚠️  Failed to clear caches: {e}")

    # ----------------------------
    # 7️⃣ Re-check Embeddings
    # ----------------------------
    try:
        embeddings2 = await manager.get_code_embeddings(sample_code)
        if embeddings2 is not None:
            print(f"\n✅ CodeBERT embeddings re-generated (dim={len(embeddings2)})")
        else:
            print("\n⚠️  CodeBERT embeddings unavailable after cache clear")
    except Exception as e:
        print(f"\n❌ Re-embedding failed: {e}")

    print("\n🎯 Demo completed successfully!\n")


if __name__ == "__main__":
    asyncio.run(main())
