import asyncio
import os
import sys
from datetime import datetime
from itertools import cycle
from colorama import Fore, Style, init
from pathlib import Path
from src.models import ModelManager
from src.rag import RAGManager
import PyPDF2

# Initialize colorama for Windows terminal support
init(autoreset=True)

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"mcp_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_to_file(content: str):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(content + "\n")

async def show_spinner(task_name: str, coro):
    spinner = cycle(["|", "/", "-", "\\"])
    done = False
    async def spin():
        while not done:
            sys.stdout.write(Fore.CYAN + f"\r{task_name}... {next(spinner)}" + Style.RESET_ALL)
            sys.stdout.flush()
            await asyncio.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(task_name) + 5) + "\r")

    spinner_task = asyncio.create_task(spin())
    result = await coro
    done = True
    await spinner_task
    print(Fore.GREEN + f"{task_name} ✅" + Style.RESET_ALL)
    return result

async def index_pdf(rag: RAGManager, pdf_path: str):
    """Extract text from PDF and index it."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    temp_file = "temp_pdf.txt"
    Path(temp_file).write_text(text, encoding="utf-8")
    await rag.index_file(temp_file, language="text")
    os.remove(temp_file)

async def main():
    print(Fore.CYAN + "🚀 Starting MCP Interactive Demo with Qwen2.5...\n" + Style.RESET_ALL)
    log_to_file("=== MCP Demo Run ===\n")

    manager = ModelManager()
    rag = RAGManager()

    # Ask user for input path
    user_path = input(Fore.YELLOW + "Enter path to code directory or PDF file: " + Style.RESET_ALL).strip()
    if not os.path.exists(user_path):
        print(Fore.RED + "Path does not exist!" + Style.RESET_ALL)
        return

    if user_path.lower().endswith(".pdf"):
        print(Fore.CYAN + f"Indexing PDF: {user_path}" + Style.RESET_ALL)
        await show_spinner("Indexing PDF", index_pdf(rag, user_path))
    elif os.path.isdir(user_path):
        print(Fore.CYAN + f"Indexing codebase at: {user_path}" + Style.RESET_ALL)
        await show_spinner("Indexing codebase", rag.index_repo(user_path))
    else:
        print(Fore.RED + "Unsupported file type. Only PDF or directories are supported." + Style.RESET_ALL)
        return

    # Ask for a query to run against RAG
    user_query = input(Fore.YELLOW + "\nEnter your query for the indexed content: " + Style.RESET_ALL).strip()
    if user_query:
        # Use general task for PDFs
        task_type = "general" if user_path.lower().endswith(".pdf") else "review"
        result = await show_spinner("Retrieving and generating response", rag.retrieve_and_generate(user_query, task=task_type))
        print(Fore.CYAN + "\n📄 Generated Response:" + Style.RESET_ALL)
        print(Fore.WHITE + result["generated_text"] + Style.RESET_ALL)
        log_to_file(f"Query: {user_query}\nResponse:\n{result['generated_text']}\n")
    else:
        print(Fore.RED + "No query provided, skipping RAG generation." + Style.RESET_ALL)

    # Optional: Code analysis if directory
    if os.path.isdir(user_path):
        py_files = list(Path(user_path).rglob("*.py"))
        if py_files:
            sample_file = py_files[0]
            code_text = sample_file.read_text(encoding="utf-8", errors="ignore")
            print(Fore.CYAN + f"\nAnalyzing sample Python file: {sample_file}" + Style.RESET_ALL)
            analysis = await show_spinner("Analyzing code", manager.analyze_code(code_text))
            print(Fore.WHITE + analysis["analysis"] + Style.RESET_ALL)
            log_to_file(f"Analysis of {sample_file}:\n{analysis['analysis']}\n")

    manager.clear_caches()
    print(Fore.GREEN + "\n🗑️  Model caches cleared successfully." + Style.RESET_ALL)
    log_to_file("Model caches cleared.\n")
    print(Fore.CYAN + "\n🎯 Demo completed successfully!\n" + Style.RESET_ALL)
    log_to_file("=== Demo completed successfully ===\n")

if __name__ == "__main__":
    asyncio.run(main())
