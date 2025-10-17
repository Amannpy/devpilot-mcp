import asyncio
import os
from datetime import datetime
from colorama import Fore, Style, init
from src.models import ModelManager

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
    print(Fore.CYAN + "🚀 Starting MCP demo with Qwen2.5 + FLAN-T5...\n" + Style.RESET_ALL)
    log_to_file("=== MCP Demo Run ===\n")

    # Example user code
    code_sample = """def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""

    print(Fore.YELLOW + "🔍 Code Sample:" + Style.RESET_ALL)
    print(Fore.WHITE + code_sample + Style.RESET_ALL)

    manager = ModelManager()

    # Run analysis
    analysis = await manager.analyze_code(code_sample)
    review = await manager.generate_review(code_sample)
    docs = await manager.generate_documentation(code_sample)
    tests = await manager.generate_tests(code_sample)

    print(Fore.CYAN + "\n💡 Code Analysis Summary:" + Style.RESET_ALL)
    print(Fore.WHITE + analysis["analysis"] + Style.RESET_ALL)
    log_to_file(f"Analysis:\n{analysis['analysis']}\n")

    print(Fore.MAGENTA + "\n📄 Generated Documentation:" + Style.RESET_ALL)
    print(Fore.WHITE + docs + Style.RESET_ALL)
    log_to_file(f"Documentation:\n{docs}\n")

    print(Fore.YELLOW + "\n🧪 Generated Unit Tests:" + Style.RESET_ALL)
    print(Fore.WHITE + tests + Style.RESET_ALL)
    log_to_file(f"Generated Tests:\n{tests}\n")

    # Clear caches
    manager.clear_caches()
    print(Fore.GREEN + "\n🗑️  Model caches cleared successfully." + Style.RESET_ALL)
    log_to_file("Model caches cleared.\n")

    print(Fore.CYAN + "\n🎯 Demo completed successfully!\n" + Style.RESET_ALL)
    log_to_file("=== Demo completed successfully ===\n")

if __name__ == "__main__":
    asyncio.run(main())
