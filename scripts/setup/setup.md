## Quick Setup Instructions

1. **Create project directory:**
```bash
mkdir mcp-ai-server
cd mcp-ai-server
```

2. **Create directory structure:**
```bash
mkdir -p src tests scripts .github/workflows
touch src/__init__.py tests/__init__.py
```

3. **Copy each file above into the appropriate location**

4. **Run setup:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

5. **Configure environment:**
```bash
# Edit .env and add your Hugging Face token
nano .env
```

6. **Run the server:**
```bash
source venv/bin/activate
python src/server.py
```

7. **Run tests:**
```bash
chmod +x scripts/run_tests.sh
./scripts/run_tests.sh
```
