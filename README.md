# DevPilot MCP : An Intelligent Developer Workflow MCP Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-1.2.0-green.svg)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered Model Context Protocol (MCP) server designed to enhance software development workflows through intelligent code review, automated documentation, bug detection, complexity analysis, and test generation. It integrates **Qwen2.5** and other Hugging Face models for high-quality AI assistance in development processes.

---

## Key Features

### MCP Tools
- **Code Review Automation** – AI-based pull request analysis with actionable feedback
- **Bug Detection** – Identifies vulnerabilities, logic issues, and common anti-patterns
- **Documentation Generation** – Automatically produces structured technical documentation
- **Complexity Analysis** – Scores code complexity and suggests refactoring options
- **Test Generation** – Generates unit tests using preferred testing frameworks

### MCP Resources
- Git repository and project health analysis
- Code quality metrics and insights
- Optional integration with issue tracking systems

### AI Models Used
- **Qwen2.5** – Advanced code understanding and generation
- **CodeBERT** – Code embedding generation
- **FLAN-T5** – Natural language generation and summarization

---

## Project Structure
```
devpilot-mcp/
├── src/
│   ├── server.py          # Core MCP server
│   ├── config.py          # Configuration and environment settings
│   ├── tools.py           # MCP tool implementations
│   ├── resources.py       # Resource definitions
│   └── models.py          # AI model integration logic
├── tests/
│   ├── test_server.py
│   ├── test_model.py
│   └── test_tools.py
├── demo.py                # Example runner for local testing
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Git
- (Optional) Hugging Face API token for extended rate limits

### Installation
```bash
git clone https://github.com/amannpy/devpilot-mcp.git
cd devpilot-mcp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set your configuration values if needed.

### Running the Server
```bash
python src/server.py
```

To try the demo script:
```bash
python demo.py
```

---

## Usage Examples

### Example 1: Code Review
```json
{
  "tool": "review_pull_request",
  "arguments": {
    "pr_content": "def calculate(a, b): return a + b",
    "language": "python"
  }
}
```

### Example 2: Bug Detection
```json
{
  "tool": "detect_bugs",
  "arguments": {
    "code_content": "query = 'SELECT * FROM users WHERE id = ' + user_id",
    "severity_filter": "critical"
  }
}
```

### Example 3: Complexity Analysis
```json
{
  "tool": "analyze_complexity",
  "arguments": {
    "code_content": "def f():\n    for i in range(10):\n        if i % 2 == 0:\n            print(i)"
  }
}
```

---

## Configuration

### Environment Variables
Example `.env`:
```bash
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxx
LOG_LEVEL=INFO
MCP_SERVER_NAME=intelligent-dev-workflow
MAX_FILE_SIZE=100000
MAX_COMPLEXITY_SCORE=10.0
```

### Advanced Settings
Edit `src/config.py` to customize:
- Model paths and APIs
- Cache strategy and expiration
- Logging and verbosity
- Complexity thresholds

---

## Testing

Run all test cases:
```bash
pytest -v
```

With coverage:
```bash
pytest --cov=src --cov-report=html
```

Individual test file:
```bash
pytest tests/test_server.py -v
```

Type checking with mypy:
```bash
mypy src/ --ignore-missing-imports
```

Linting with ruff:
```bash
ruff check
ruff check --fix  # Auto-fix issues
```

---

## MCP Integration

Example configuration for an MCP client:
```json
{
  "mcpServers": {
    "intelligent-dev-workflow": {
      "command": "python",
      "args": ["src/server.py"],
      "env": {
        "HUGGINGFACE_API_TOKEN": "your_token_here"
      }
    }
  }
}
```

---

## Available Tools

| Tool Name | Description | Input Parameters |
|-----------|-------------|------------------|
| `review_pull_request` | AI code review | `pr_content`, `language` |
| `generate_documentation` | Create docs | `code_content`, `doc_style` |
| `detect_bugs` | Detect vulnerabilities | `code_content`, `severity_filter` |
| `analyze_complexity` | Analyze complexity | `code_content` |
| `generate_tests` | Generate unit tests | `code_content`, `test_framework` |

---

## Performance and Design

- **Caching**: In-memory caching with configurable TTL
- **Asynchronous Processing**: Non-blocking async I/O using asyncio
- **Rate Limiting**: Adaptive throttling for API usage
- **Logging**: Structured JSON and console logging options

---

## Development Guidelines

- Follows PEP 8 coding standards
- Uses type hints throughout (mypy compatible)
- Includes unit tests for all core modules
- Well-documented, modular architecture

To contribute:
```bash
git checkout -b feature/your-feature
git commit -m "Add new feature"
git push origin feature/your-feature
```

Then open a Pull Request.

---

## License

Licensed under the MIT License.

---

## Contact

- **GitHub Issues**: [Open an issue](https://github.com/Amannpy/mcp-ai-server/issues)
- **Discussions**: [Join discussion](https://github.com/Amannpy/mcp-ai-server/discussions)
- **Email**: aman.kumar.cse2611@gmail.com

---

## Roadmap

- [ ] GitHub Actions CI/CD enhancements
- [ ] VS Code and JetBrains plugin integration
- [ ] Real-time web dashboard
- [ ] Expanded multi-language model support
- [ ] SaaS deployment template

---

**Developed for modern developers seeking to integrate AI intelligence into their workflow.**