# 📘 Intelligent Developer Workflow MCP Server - Usage Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Tool Reference](#tool-reference)
3. [Best Practices](#best-practices)
4. [Advanced Usage](#advanced-usage)
5. [Integration Examples](#integration-examples)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Run Checklist

```bash
# 1. Verify Python version
python --version  # Should be 3.10+

# 2. Activate virtual environment
source venv/bin/activate

# 3. Verify installation
pip list | grep mcp

# 4. Test server
python src/server.py
```

### Quick Test

```python
# test_connection.py
import asyncio
import json

async def test_server():
    # This would use MCP client to connect
    print("Testing MCP server connection...")
    
if __name__ == "__main__":
    asyncio.run(test_server())
```

---

## Tool Reference

### 1. Code Review Tool

#### Purpose
Analyzes pull requests and provides AI-powered suggestions for code improvements.

#### Parameters
```json
{
  "pr_content": "string (required) - Code diff or full code content",
  "language": "string (optional) - Programming language (default: python)"
}
```

#### Supported Languages
- Python
- JavaScript/TypeScript
- Java
- C++
- Go
- Rust
- Ruby

#### Example Usage

**Basic Review:**
```python
{
  "tool": "review_pull_request",
  "arguments": {
    "pr_content": """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
    """,
    "language": "python"
  }
}
```

**Response:**
```json
{
  "timestamp": "2025-10-13T14:30:00",
  "language": "python",
  "static_issues": [
    {
      "type": "performance",
      "severity": "low",
      "message": "Consider using list comprehension for better performance"
    }
  ],
  "ai_suggestions": [
    "Add type hints for function parameters and return value",
    "Consider edge case: empty data list",
    "Add docstring explaining function purpose"
  ],
  "overall_score": 75.5,
  "summary": "Code is functional with room for improvements in documentation and type safety"
}
```

**Advanced Review with Context:**
```python
# Review a full pull request
{
  "tool": "review_pull_request",
  "arguments": {
    "pr_content": """
+ class UserService:
+     def __init__(self, db_connection):
+         self.db = db_connection
+ 
+     def create_user(self, username, password):
+         query = "INSERT INTO users VALUES ('" + username + "', '" + password + "')"
+         self.db.execute(query)
+         return True
    """,
    "language": "python"
  }
}
```

**Response will flag security issues:**
```json
{
  "static_issues": [
    {
      "type": "sql_injection",
      "severity": "critical",
      "message": "SQL injection vulnerability detected"
    }
  ],
  "ai_suggestions": [
    "Use parameterized queries to prevent SQL injection",
    "Hash passwords before storing",
    "Add input validation",
    "Implement proper error handling"
  ],
  "overall_score": 25.0
}
```

---

### 2. Bug Detection Tool

#### Purpose
Identifies security vulnerabilities, code smells, and potential bugs through pattern matching and static analysis.

#### Parameters
```json
{
  "code_content": "string (required) - Source code to analyze",
  "severity_filter": "string (optional) - Filter by severity: all|critical|high|medium|low"
}
```

#### Detection Categories

**Security Bugs:**
- SQL Injection
- Command Injection
- Path Traversal
- Hardcoded Credentials
- Unsafe Deserialization

**Code Quality:**
- Bare except clauses
- Debug print statements
- Eval usage
- Large functions
- High complexity

#### Example Usage

**Security Scan:**
```python
{
  "tool": "detect_bugs",
  "arguments": {
    "code_content": """
import os
import pickle

def load_config(filename):
    with open(filename, 'rb') as f:
        config = pickle.load(f)
    return config

def run_command(user_input):
    os.system('ping ' + user_input)

PASSWORD = 'admin123'
    """,
    "severity_filter": "critical"
  }
}
```

**Response:**
```json
{
  "bugs_found": 2,
  "bugs": [
    {
      "type": "command_injection",
      "severity": "critical",
      "line": 9,
      "snippet": "os.system('ping ' + user_input)",
      "description": "Unsanitized user input in system command"
    },
    {
      "type": "hardcoded_password",
      "severity": "high",
      "line": 11,
      "snippet": "PASSWORD = 'admin123'",
      "description": "Hardcoded credentials detected"
    }
  ],
  "severity_filter": "critical",
  "analyzed_at": "2025-10-13T14:30:00"
}
```

---

### 3. Documentation Generation Tool

#### Purpose
Automatically generates comprehensive technical documentation from source code.

#### Parameters
```json
{
  "code_content": "string (required) - Code to document",
  "doc_style": "string (optional) - Format: markdown|restructuredtext|docstring"
}
```

#### Documentation Styles

**Markdown (default):**
```python
{
  "tool": "generate_documentation",
  "arguments": {
    "code_content": """
class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
    """,
    "doc_style": "markdown"
  }
}
```

**Response:**
```json
{
  "documentation": "# Calculator\n\nA simple calculator class for basic arithmetic operations.\n\n## Methods\n\n### add(a, b)\nAdds two numbers and returns the result.\n\n**Parameters:**\n- `a`: First number\n- `b`: Second number\n\n**Returns:** Sum of a and b\n\n### multiply(a, b)\nMultiplies two numbers and returns the result.\n\n**Parameters:**\n- `a`: First number\n- `b`: Second number\n\n**Returns:** Product of a and b",
  "style": "markdown",
  "entities_documented": ["Calculator", "add", "multiply"],
  "generated_at": "2025-10-13T14:30:00"
}
```

---

### 4. Complexity Analysis Tool

#### Purpose
Analyzes code complexity and provides refactoring recommendations.

#### Parameters
```json
{
  "code_content": "string (required) - Code to analyze"
}
```

#### Metrics Calculated
- Lines of Code (LOC)
- Cyclomatic Complexity
- Function Count
- Class Count
- Maximum Nesting Level
- Maintainability Index

#### Example Usage

**Simple Code:**
```python
{
  "tool": "analyze_complexity",
  "arguments": {
    "code_content": """
def simple_function(x):
    return x * 2

class SimpleClass:
    pass
    """
  }
}
```

**Response:**
```json
{
  "complexity_score": 15.2,
  "metrics": {
    "lines_of_code": 5,
    "functions": 1,
    "classes": 1,
    "max_nesting_level": 1
  },
  "recommendations": [
    "Code complexity is acceptable"
  ]
}
```

**Complex Code:**
```python
{
  "tool": "analyze_complexity",
  "arguments": {
    "code_content": """
def complex_function(data):
    results = []
    for item in data:
        if item['status'] == 'active':
            if item['priority'] > 5:
                if item['type'] == 'urgent':
                    for sub_item in item['details']:
                        if sub_item['valid']:
                            results.append(sub_item)
    return results
    """
  }
}
```

**Response:**
```json
{
  "complexity_score": 78.5,
  "metrics": {
    "lines_of_code": 10,
    "functions": 1,
    "classes": 0,
    "max_nesting_level": 5
  },
  "recommendations": [
    "High complexity detected",
    "Refactor into smaller functions",
    "Extract complex logic into separate functions",
    "Consider using filter() and comprehensions"
  ]
}
```

---

### 5. Test Generation Tool

#### Purpose
Generates unit tests automatically based on source code.

#### Parameters
```json
{
  "code_content": "string (required) - Code to generate tests for",
  "test_framework": "string (optional) - Framework: pytest|unittest|jest"
}
```

#### Example Usage

**Pytest Generation:**
```python
{
  "tool": "generate_tests",
  "arguments": {
    "code_content": """
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
    """,
    "test_framework": "pytest"
  }
}
```

**Response:**
```json
{
  "test_code": "import pytest\nfrom module import divide\n\ndef test_divide_basic():\n    assert divide(10, 2) == 5.0\n\ndef test_divide_zero():\n    with pytest.raises(ValueError):\n        divide(10, 0)\n\ndef test_divide_negative():\n    assert divide(-10, 2) == -5.0\n\ndef test_divide_floats():\n    assert abs(divide(7, 3) - 2.333) < 0.01",
  "framework": "pytest",
  "generated_at": "2025-10-13T14:30:00"
}
```

---

## Best Practices

### 1. Code Review Workflow

```python
# Recommended workflow for PR reviews

# Step 1: Run bug detection first
bugs = detect_bugs(pr_content, severity_filter="critical")
if bugs['bugs_found'] > 0:
    print("⚠️ Critical bugs found! Fix before proceeding.")
    exit(1)

# Step 2: Analyze complexity
complexity = analyze_complexity(pr_content)
if complexity['complexity_score'] > 60:
    print("⚠️ High complexity detected")

# Step 3: Run full review
review = review_pull_request(pr_content, language="python")

# Step 4: Generate documentation if needed
if review['overall_score'] > 70:
    docs = generate_documentation(pr_content)
```

### 2. Caching Strategy

The server caches responses for identical requests. To optimize:

```python
# ✅ Good: Consistent formatting
code1 = "def test(): pass"
code2 = "def test(): pass"  # Cache hit

# ❌ Bad: Different formatting
code1 = "def test(): pass"
code2 = "def test():    pass"  # Cache miss
```

### 3. Rate Limiting

With Hugging Face free tier:
- 100 requests per hour
- Batch your requests
- Use caching effectively

```python
# ✅ Good: Batch processing
for file in files[:10]:  # Process in batches
    review = review_pull_request(file)
    time.sleep(1)  # Rate limiting

# ❌ Bad: Rapid fire requests
for file in files:  # May hit rate limit
    review = review_pull_request(file)
```

### 4. Error Handling

```python
# ✅ Proper error handling
try:
    result = review_pull_request(code, "python")
    if 'error' in result:
        print(f"Error: {result['error']}")
        # Fallback to static analysis only
except Exception as e:
    print(f"Server error: {e}")
    # Implement retry logic
```

---

## Advanced Usage

### 1. Custom MCP Client

```python
import asyncio
import json
from mcp.client import Client

async def advanced_review():
    async with Client() as client:
        # Connect to server
        await client.connect("stdio", ["python", "src/server.py"])
        
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")
        
        # Call tool
        result = await client.call_tool(
            "review_pull_request",
            {"pr_content": "...", "language": "python"}
        )
        
        return json.loads(result[0].text)

asyncio.run(advanced_review())
```

### 2. Batch Processing

```python
async def batch_review(file_paths):
    results = []
    
    for path in file_paths:
        with open(path) as f:
            code = f.read()
        
        # Review
        review = await review_pull_request(code, "python")
        
        # Bug detection
        bugs = await detect_bugs(code, "all")
        
        # Complexity
        complexity = await analyze_complexity(code)
        
        results.append({
            'file': path,
            'review': review,
            'bugs': bugs,
            'complexity': complexity
        })
    
    return results
```

### 3. CI/CD Integration

```yaml
# .github/workflows/code-review.yml
name: AI Code Review

on: [pull_request]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install MCP Server
        run: pip install -r requirements.txt
      
      - name: Run AI Review
        run: |
          python scripts/ci_review.py \
            --files-changed ${{ github.event.pull_request.changed_files }}
      
      - name: Post Results
        uses: actions/github-script@v6
        with:
          script: |
            // Post review comments to PR
```

### 4. Custom Analysis Pipeline

```python
class CodeAnalysisPipeline:
    """Custom analysis pipeline"""
    
    def __init__(self, server):
        self.server = server
        self.results = {}
    
    async def analyze(self, code, language="python"):
        # Stage 1: Security
        self.results['security'] = await self.server.detect_bugs(
            code, 
            severity_filter="critical"
        )
        
        if self.results['security']['bugs_found'] > 0:
            return self.results  # Stop on critical bugs
        
        # Stage 2: Complexity
        self.results['complexity'] = await self.server.analyze_complexity(code)
        
        # Stage 3: Review
        self.results['review'] = await self.server.review_pull_request(
            code, 
            language
        )
        
        # Stage 4: Documentation
        if self.results['review']['overall_score'] > 70:
            self.results['docs'] = await self.server.generate_documentation(code)
        
        return self.results
    
    def generate_report(self):
        """Generate markdown report"""
        report = "# Code Analysis Report\n\n"
        
        # Security section
        report += "## Security\n"
        bugs = self.results['security']['bugs_found']
        report += f"Bugs found: {bugs}\n\n"
        
        # Complexity section
        report += "## Complexity\n"
        score = self.results['complexity']['complexity_score']
        report += f"Score: {score}/100\n\n"
        
        # Review section
        report += "## Review\n"
        overall = self.results['review']['overall_score']
        report += f"Overall score: {overall}/100\n"
        
        return report
```

---

## Integration Examples

### 1. VS Code Extension

```typescript
// extension.ts
import * as vscode from 'vscode';
import { MCPClient } from 'mcp-client';

export function activate(context: vscode.ExtensionContext) {
    const client = new MCPClient();
    
    // Register command
    let disposable = vscode.commands.registerCommand(
        'mcp.reviewCode',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;
            
            const code = editor.document.getText();
            const language = editor.document.languageId;
            
            // Call MCP server
            const result = await client.callTool('review_pull_request', {
                pr_content: code,
                language: language
            });
            
            // Show results
            vscode.window.showInformationMessage(
                `Code score: ${result.overall_score}/100`
            );
        }
    );
    
    context.subscriptions.push(disposable);
}
```

### 2. Git Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running AI code review..."

# Get staged Python files
FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '.py)

if [ -z "$FILES" ]; then
    exit 0
fi

# Run review
python scripts/pre_commit_review.py $FILES

if [ $? -ne 0 ]; then
    echo "❌ Code review failed. Fix issues before committing."
    exit 1
fi

echo "✅ Code review passed!"
exit 0
```

### 3. Slack Bot

```python
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.command("/review")
async def review_command(ack, command, say):
    await ack()
    
    code = command['text']
    
    # Run review
    result = await review_pull_request(code, "python")
    
    # Format response
    message = f"""
    *Code Review Results*
    Score: {result['overall_score']}/100
    Issues: {len(result['static_issues'])}
    Suggestions: {len(result['ai_suggestions'])}
    """
    
    await say(message)

handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
handler.start()
```

---

## Troubleshooting

### Common Issues

#### 1. API Rate Limiting
```
Error: Rate limit exceeded
```

**Solution:**
```python
# Add HUGGINGFACE_API_TOKEN to .env
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxx

# Or implement backoff
import time

def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except RateLimitError:
            time.sleep(2 ** i)
    raise Exception("Max retries exceeded")
```

#### 2. Large File Processing
```
Error: Input too large
```

**Solution:**
```python
# Split large files
def chunk_code(code, max_lines=500):
    lines = code.split('\n')
    for i in range(0, len(lines), max_lines):
        yield '\n'.join(lines[i:i+max_lines])

# Process in chunks
for chunk in chunk_code(large_file):
    result = review_pull_request(chunk, "python")
```

#### 3. Connection Errors
```
Error: Failed to connect to server
```

**Solution:**
```python
# Check server is running
ps aux | grep server.py

# Verify stdio connection
python -c "from mcp.server.stdio import stdio_server; print('OK')"

# Check logs
tail -f logs/mcp-server.log
```

---

## Performance Tips

### 1. Optimize for Speed
```python
# Use async for multiple files
async def fast_review(files):
    tasks = [review_pull_request(f, "python") for f in files]
    return await asyncio.gather(*tasks)
```

### 2. Cache Aggressively
```python
# Pre-compute for common patterns
COMMON_PATTERNS = {
    "empty_function": "def test(): pass",
    "hello_world": "print('hello')"
}

# Cache these at startup
for pattern in COMMON_PATTERNS.values():
    review_pull_request(pattern, "python")
```

### 3. Monitor Performance
```python
import time

def timed_review(code, language):
    start = time.time()
    result = review_pull_request(code, language)
    elapsed = time.time() - start
    
    print(f"Review took {elapsed:.2f}s")
    return result
```

---

## Support & Resources

- **Documentation**: [Full API Reference](docs/API.md)
- **Examples**: [Example Scripts](examples/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/mcp-ai-server/issues)
- **Community**: [Discussions](https://github.com/yourusername/mcp-ai-server/discussions)

---

**Happy Coding! 🚀**