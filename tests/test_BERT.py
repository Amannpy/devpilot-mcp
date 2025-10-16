import asyncio
from src.models import CodeBERTModel

async def test_real_codebert():
    model = CodeBERTModel()

    # A small, simple code snippet
    code = """
def multiply(a, b):
    return a * b
"""

    print("Running real CodeBERT inference...")
    result = await model.analyze_code(code)
    print("✅ CodeBERT result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_real_codebert())
