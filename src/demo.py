import asyncio
import os
from datetime import datetime
from colorama import Fore, Style, init
from models import analyze_code, clear_model_caches

# Initialize colorama for Windows terminal support
init(autoreset=True)

# Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"mcp_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_to_file(content: str):
    """Append content to the log file."""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(content + "\n")

async def main():
    print(Fore.CYAN + "🚀 Starting MCP demo with CodeBERT + FLAN-T5...\n" + Style.RESET_ALL)
    log_to_file("=== MCP Demo Run ===\n")

    # Example user code to test both models
    code_sample = """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
    print(Fore.YELLOW + "🔍 Code Analysis:" + Style.RESET_ALL)
    print(Fore.WHITE + code_sample + Style.RESET_ALL)

    result = analyze_code(code_sample)

    if result["embeddings"] is not None:
        print(Fore.GREEN + f"\n✅ CodeBERT embeddings generated (dim={result['embeddings'].shape[1]})" + Style.RESET_ALL)
        log_to_file("Embeddings generated successfully.")
    else:
        print(Fore.RED + "\n⚠️ CodeBERT embeddings failed" + Style.RESET_ALL)
        log_to_file("Embeddings failed.")

    print(Fore.CYAN + "\n💡 FLAN-T5 Code Review:" + Style.RESET_ALL)
    print(Fore.WHITE + result["review"] + Style.RESET_ALL)
    log_to_file(f"Code Review:\n{result['review']}\n")

    print(Fore.MAGENTA + "\n📄 FLAN-T5 Documentation:" + Style.RESET_ALL)
    print(Fore.WHITE + result["docs"] + Style.RESET_ALL)
    log_to_file(f"Documentation:\n{result['docs']}\n")

    print(Fore.YELLOW + "\n🧪 FLAN-T5 Generated Tests:" + Style.RESET_ALL)
    print(Fore.WHITE + result["tests"] + Style.RESET_ALL)
    log_to_file(f"Generated Tests:\n{result['tests']}\n")

    # Clear caches at the end
    clear_model_caches()
    print(Fore.GREEN + "\n🗑️  Model caches cleared successfully." + Style.RESET_ALL)
    log_to_file("Model caches cleared.\n")

    if result["embeddings"] is not None:
        print(Fore.GREEN + f"\n✅ CodeBERT embeddings re-generated (dim={result['embeddings'].shape[1]})" + Style.RESET_ALL)
    else:
        print(Fore.RED + "\n⚠️ CodeBERT re-generation failed" + Style.RESET_ALL)

    print(Fore.CYAN + "\n🎯 Demo completed successfully!\n" + Style.RESET_ALL)
    log_to_file("=== Demo completed successfully ===\n")

if __name__ == "__main__":
    asyncio.run(main())
