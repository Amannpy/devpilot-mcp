#!/bin/bash
# Setup script for the project

echo "Setting up MCP AI Developer Server..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please add your HUGGINGFACE_API_TOKEN"
fi

# Create necessary directories
mkdir -p logs
mkdir -p cache
mkdir -p tests

echo "Setup complete! Activate the environment with: source venv/bin/activate"
```

---

## File 14: `scripts/run_tests.sh`

```bash
#!/bin/bash
# Run comprehensive test suite

echo "Running MCP AI Server Test Suite..."

# Activate virtual environment
source venv/bin/activate

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run security checks
bandit -r src/ -ll

# Run type checking
mypy src/ --ignore-missing-imports

echo "Tests complete! Check htmlcov/index.html for coverage report"
