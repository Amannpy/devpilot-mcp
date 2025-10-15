from src.server import DeveloperWorkflowServer
import asyncio

async def main():
    server = DeveloperWorkflowServer()
    print("✅ Server initialized.")

    # --- Test complexity analysis ---
    code_snippet = """
def foo(x):
    if x > 5:
        print("hi")
    return x
"""
    complexity_result = await server.analyze_complexity(code_snippet)
    print("🔍 analyze_complexity output:")
    print(complexity_result)

    # --- Test bug detection ---
    buggy_code = 'password = "1234"\nprint("debug")'
    bug_result = await server.detect_bugs(buggy_code, 'all')
    print("🐞 detect_bugs output:")
    print(bug_result)

    # --- Test documentation generation (mock HF call) ---
    # Temporarily disable HF inference for local testing
    server.hf_client.text_generation = lambda *args, **kwargs: "Mocked documentation output"
    doc_result = await server.generate_documentation("def add(a, b): return a + b", "markdown")
    print("📘 generate_documentation output:")
    print(doc_result)

    # --- Test test-generation (mock HF call) ---
    test_result = await server.generate_tests("def add(a, b): return a + b", "pytest")
    print("🧪 generate_tests output:")
    print(test_result)

asyncio.run(main())
