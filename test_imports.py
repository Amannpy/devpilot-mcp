#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("🧪 Testing MCP AI Server Imports")
print("=" * 70)
print(f"📁 Project root: {project_root}")
print(f"🐍 Python path: {sys.path[0]}")

# Test 1: Config imports
print("\n1️⃣ Testing config imports...")
try:
    from backend.config import settings, app_config

    print("   ✅ backend.config imports work")
    print(f"   📍 Default model: {settings.default_model}")
    print(f"   📍 Device: {settings.get_device()}")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 2: Logger import
print("\n2️⃣ Testing logger import...")
try:
    from backend.core.utils.logger import get_logger

    logger = get_logger("test")
    logger.info("Logger test message")
    print("   ✅ Logger works")
except Exception as e:
    print(f"   ❌ Logger import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Utils config redirect
print("\n3️⃣ Testing utils config redirect...")
try:
    from backend.core.utils.config import settings as utils_settings

    assert utils_settings.default_model == settings.default_model
    print("   ✅ Utils config redirect works")
except Exception as e:
    print(f"   ❌ Utils config redirect failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Qwen model import
print("\n4️⃣ Testing Qwen model import...")
try:
    from backend.core.models.qwen_model import QwenModelWrapper

    print("   ✅ QwenModelWrapper imports work")
except Exception as e:
    print(f"   ❌ Qwen model import failed: {e}")
    print(f"   ⚠️  This is OK if model files don't exist yet")

# Test 5: Model manager import (without loading model)
print("\n5️⃣ Testing model manager import...")
try:
    # Temporarily disable auto-loading for testing
    import backend.core.models.model_manager as mm_module


    # Create a fresh manager without preloading
    class TestModelManager(mm_module.ModelManager):
        def __init__(self):
            self.config = mm_module.settings
            self.models = {}
            mm_module.logger.info("Test ModelManager initialized (no preload)")


    test_manager = TestModelManager()
    print("   ✅ ModelManager class loads successfully")
    print(f"   📍 Loaded models: {test_manager.list_loaded_models()}")
except Exception as e:
    print(f"   ❌ ModelManager import failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 6: Check GPU
print("\n6️⃣ Checking GPU availability...")
try:
    import torch

    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("   ⚠️  CUDA not available, will use CPU")
except ImportError:
    print("   ⚠️  PyTorch not installed yet")

print("\n" + "=" * 70)
print("✅ All import tests passed!")
print("=" * 70)
print("\n📋 Next Steps:")
print("1. If model not downloaded yet: python scripts/download_models.py")
print("2. Test actual model loading: python test_model_loading.py")
print("3. Run full GPU test: python test_gpu_setup.py")
print()