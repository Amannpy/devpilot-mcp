# 🚀 Intelligent Developer Workflow MCP Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP Protocol](https://img.shields.io/badge/MCP-1.2.0-green.svg)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered Model Context Protocol (MCP) server that integrates with development workflows, providing intelligent code review, documentation generation, bug detection, and project management capabilities using Hugging Face models.

## 🎯 Key Features

### MCP Tools
- **🔍 Code Review Automation** - AI-powered pull request reviews with actionable suggestions
- **🐛 Bug Detection** - Static analysis combined with AI to identify security vulnerabilities and code smells
- **📚 Documentation Generation** - Automatic technical documentation from source code
- **📊 Complexity Analysis** - Code complexity scoring with refactoring recommendations
- **🧪 Test Generation** - Auto-generate unit tests for your code

### MCP Resources
- **Git Repository Analysis** - Analyze repository structure and health metrics
- **Project Metrics** - Code quality metrics and statistics dashboard
- **Issue Tracking Integration** - Connect with popular issue tracking systems

### AI Models
- **microsoft/CodeBERT-base** - For code understanding and analysis
- **google/flan-t5-base** - For natural language generation and suggestions

## 🏗️ Architecture

```
mcp-ai-server/
├── src/
│   ├── server.py           # Main MCP server implementation
│   ├── config.py           # Configuration management
│   ├── tools/              # MCP tool implementations
│   ├── resources/          # MCP resource handlers
│   └── models/             # AI model integrations
├── tests/
│   ├── test_server.py      # Comprehensive test suite
│   └── fixtures/           # Test fixtures
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── pyproject.toml         # Project metadata
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- PyCharm (recommended) or any Python IDE
- Git
- Hugging Face account (optional, for higher API limits)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mcp-ai-server.git
cd mcp-ai-server
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your Hugging Face API token (optional)
```

5. **Run the server**
```bash
python src/server.py
```

### PyCharm Setup

1. Open PyCharm and select "Open" → Navigate to project directory
2. Configure Python interpreter: File → Settings → Project → Python Interpreter
3. Select the virtual environment created above
4. Install requirements through PyCharm's package manager
5. Create run configuration:
   - Script path: `src/server.py`
   - Working directory: project root
   - Environment variables: Load from `.env`

## 📖 Usage Examples

### Code Review

```python
# Using the MCP protocol
{
  "tool": "review_pull_request",
  "arguments": {
    "pr_content": "def calculate(a, b):\n    return a + b",
    "language": "python"
  }
}
```

Response:
```json
{
  "timestamp": "2025-10-13T10:30:00",
  "language": "python",
  "static_issues": [],
  "ai_suggestions": [
    "Add type hints for better code clarity",
    "Include docstring with parameter descriptions"
  ],
  "overall_score": 85.0,
  "summary": "Code looks good with minor improvements suggested"
}
```

### Bug Detection

```python
{
  "tool": "detect_bugs",
  "arguments": {
    "code_content": "query = 'SELECT * FROM users WHERE id = ' + user_id",
    "severity_filter": "critical"
  }
}
```

Response:
```json
{
  "bugs_found": 1,
  "bugs": [
    {
      "type": "sql_injection",
      "severity": "critical",
      "line": 1,
      "snippet": "query = 'SELECT * FROM users WHERE id = ' + user_id"
    }
  ]
}
```

### Documentation Generation

```python
{
  "tool": "generate_documentation",
  "arguments": {
    "code_content": "class UserManager:\n    def create_user(self, name, email):\n        pass",
    "doc_style": "markdown"
  }
}
```

### Complexity Analysis

```python
{
  "tool": "analyze_complexity",
  "arguments": {
    "code_content": "your_code_here"
  }
}
```

Response:
```json
{
  "complexity_score": 45.2,
  "metrics": {
    "lines_of_code": 150,
    "functions": 12,
    "classes": 3,
    "max_nesting_level": 4
  },
  "recommendations": [
    "Consider breaking down large functions",
    "Review nesting levels"
  ]
}
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Hugging Face API (optional - for higher rate limits)
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Server Configuration
LOG_LEVEL=INFO
MCP_SERVER_NAME=intelligent-dev-workflow

# Analysis Settings
MAX_FILE_SIZE=100000
MAX_COMPLEXITY_SCORE=10.0
```

### Advanced Configuration

Edit `src/config.py` to customize:
- Model selection
- Rate limiting
- Cache settings
- Analysis thresholds

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test class
pytest tests/test_server.py::TestCodeReview -v

# Run with detailed output
pytest tests/ -vv -s
```

Test coverage includes:
- Code review functionality
- Bug detection patterns
- Documentation generation
- Complexity analysis
- Error handling
- Integration workflows

## 🎨 Features in Detail

### 1. AI-Powered Code Review
- Combines static analysis with AI suggestions
- Identifies code smells and anti-patterns
- Provides actionable improvement recommendations
- Caches results for better performance

### 2. Security Bug Detection
Detects common vulnerabilities:
- SQL injection risks
- Hardcoded credentials
- Unsafe eval() usage
- Exception handling issues
- Debug code in production

### 3. Documentation Generation
- Generates comprehensive documentation
- Supports multiple formats (Markdown, reStructuredText, Docstrings)
- Extracts function and class signatures
- Includes usage examples

### 4. Complexity Analysis
- Calculates cyclomatic complexity
- Tracks nesting depth
- Counts functions and classes
- Provides refactoring suggestions

### 5. Test Generation
- Generates unit tests automatically
- Supports multiple frameworks (pytest, unittest, jest)
- Includes edge cases and error scenarios

## 🔌 MCP Integration

### Using with MCP Clients

The server implements the Model Context Protocol and can be used with any MCP-compatible client:

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

### Available MCP Tools

| Tool Name | Description | Input Parameters |
|-----------|-------------|------------------|
| `review_pull_request` | AI code review | `pr_content`, `language` |
| `generate_documentation` | Generate docs | `code_content`, `doc_style` |
| `detect_bugs` | Find vulnerabilities | `code_content`, `severity_filter` |
| `analyze_complexity` | Complexity analysis | `code_content` |
| `generate_tests` | Create unit tests | `code_content`, `test_framework` |

### Available MCP Resources

| Resource URI | Description |
|--------------|-------------|
| `git://repository/analysis` | Repository health metrics |
| `project://metrics/overview` | Project quality statistics |

## 🚀 Performance Optimization

### Caching Strategy
- Response caching for identical requests
- Configurable cache size and TTL
- In-memory storage for fast access

### Rate Limiting
- 100 requests per hour (default)
- Respects Hugging Face API limits
- Graceful degradation on rate limit

### Async Processing
- Fully asynchronous architecture
- Concurrent request handling
- Non-blocking I/O operations

## 📊 Metrics & Monitoring

The server tracks:
- Request count and latency
- Cache hit rates
- Model inference times
- Error rates by type

Access metrics via:
```python
# Resource endpoint
GET project://metrics/overview
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure all tests pass

## 🐛 Troubleshooting

### Common Issues

**Issue: Hugging Face API errors**
```bash
Solution: Set HUGGINGFACE_API_TOKEN in .env file
```

**Issue: Import errors**
```bash
Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**Issue: Server not starting**
```bash
Solution: Check Python version (3.10+) and port availability
python --version
```

## 📚 Resources

- [MCP Protocol Documentation](https://modelcontextprotocol.io)
- [Hugging Face Models](https://huggingface.co/models)
- [CodeBERT Paper](https://arxiv.org/abs/2002.08155)
- [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/mcp-ai-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/mcp-ai-server/discussions)
- **Email**: your.email@example.com

## 🎓 LinkedIn Impact

This project demonstrates:
- ✅ **MCP Protocol Implementation** - Understanding of modern AI integration patterns
- ✅ **AI/ML Integration** - Practical use of Hugging Face models
- ✅ **Developer Tools** - Building productivity-enhancing tools
- ✅ **Software Architecture** - Clean, maintainable, async design
- ✅ **Testing & Quality** - Comprehensive test coverage
- ✅ **Documentation** - Professional project documentation

Perfect for showcasing on your LinkedIn profile as:
- Portfolio project demonstrating AI skills
- Open-source contribution
- Technical blog post topic
- Case study for interviews

## 🗺️ Roadmap

- [ ] GitHub Actions integration
- [ ] GitLab CI/CD support
- [ ] VS Code extension
- [ ] Real-time collaboration features
- [ ] Custom model fine-tuning
- [ ] Multi-language support expansion
- [ ] Web dashboard interface

## 🙏 Acknowledgments

- Anthropic for the MCP Protocol
- Hugging Face for model hosting
- Microsoft for CodeBERT
- Google for FLAN-T5

---

**Made with ❤️ for developers by developers**

*Star ⭐ this repo if you find it useful!*