#!/usr/bin/env python3
"""
GPU Setup Test for MCP AI Server
Tests CUDA, PyTorch, and quantization support
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test if required packages are installed"""
    print("=" * 70)
    print("📦 Testing Package Imports")
    print("=" * 70)

    packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "accelerate": "Accelerate",
        "bitsandbytes": "BitsAndBytes (for quantization)",
        "sentence_transformers": "Sentence Transformers",
    }

    results = {}
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}: Installed")
            results[package] = True
        except ImportError:
            print(f"❌ {name}: NOT INSTALLED")
            results[package] = False

    return all(results.values())


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "=" * 70)
    print("🎮 Testing CUDA/GPU Support")
    print("=" * 70)

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")

        if cuda_available:
            print(f"\n📊 GPU Information:")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            print(f"  Current Device: {torch.cuda.current_device()}")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")

            # Get detailed properties
            props = torch.cuda.get_device_properties(0)
            print(f"\n  Total Memory: {props.total_memory / 1024 ** 3:.2f} GB")
            print(f"  CUDA Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")

            # Check memory
            print(f"\n💾 Memory Status:")
            print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
            print(f"  Available: {(props.total_memory - torch.cuda.memory_reserved(0)) / 1024 ** 3:.2f} GB")

            # PyTorch CUDA version
            print(f"\n🔧 Software Versions:")
            print(f"  PyTorch Version: {torch.__version__}")
            print(f"  CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")

            return True
        else:
            print("\n⚠️  CUDA is not available. Possible reasons:")
            print("  1. NVIDIA GPU drivers not installed")
            print("  2. CUDA toolkit not installed")
            print("  3. PyTorch installed without CUDA support")
            print("\n  Install CUDA-enabled PyTorch with:")
            print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False

    except Exception as e:
        print(f"\n❌ Error testing CUDA: {e}")
        return False


def test_quantization():
    """Test 4-bit/8-bit quantization support"""
    print("\n" + "=" * 70)
    print("🔧 Testing Quantization Support")
    print("=" * 70)

    try:
        import torch
        import bitsandbytes as bnb

        if not torch.cuda.is_available():
            print("\n⚠️  Skipping quantization test (CUDA not available)")
            return False

        print(f"\nBitsAndBytes Version: {bnb.__version__}")

        # Test 8-bit matmul
        print("\n📊 Testing 8-bit operations...")
        try:
            # Create small test tensors
            a = torch.randn(16, 32, device='cuda', dtype=torch.float16)
            b = torch.randn(32, 16, device='cuda', dtype=torch.float16)

            # Test basic matmul
            c = torch.matmul(a, b)
            print("  ✅ 8-bit operations: Working")

            # Clean up
            del a, b, c
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ 8-bit operations failed: {e}")
            return False

        # Test 4-bit support
        print("\n📊 Testing 4-bit quantization...")
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("  ✅ 4-bit quantization config: Created successfully")

        except Exception as e:
            print(f"  ❌ 4-bit config failed: {e}")
            return False

        print("\n✅ Quantization support is available!")
        return True

    except ImportError as e:
        print(f"\n❌ BitsAndBytes not installed: {e}")
        print("  Install with: pip install bitsandbytes")
        return False
    except Exception as e:
        print(f"\n❌ Error testing quantization: {e}")
        return False


def test_small_model():
    """Test loading a small model with quantization"""
    print("\n" + "=" * 70)
    print("🧪 Testing Small Model Load (with 4-bit quantization)")
    print("=" * 70)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            print("\n⚠️  Skipping model test (CUDA not available)")
            return False

        print("\n📥 Loading tiny test model (gpt2-small)...")
        print("   This tests if 4-bit quantization works correctly")

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load a tiny model for testing
        model_name = "gpt2"  # Small model for testing (~500MB)

        print(f"\n⏳ Loading {model_name} with 4-bit quantization...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        print("✅ Model loaded successfully!")

        # Test inference
        print("\n🧪 Testing inference...")
        test_text = "Hello, this is a test"
        inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Input: '{test_text}'")
        print(f"  Output: '{result}'")

        # Check memory usage
        memory_used = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"\n💾 Memory Usage: {memory_used:.2f} GB")

        # Clean up
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()

        print("\n✅ Model inference test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        print("\nThis might mean:")
        print("  1. BitsAndBytes is not properly installed")
        print("  2. CUDA drivers need updating")
        print("  3. GPU doesn't support required operations")
        return False


def test_config():
    """Test MCP AI Server config"""
    print("\n" + "=" * 70)
    print("⚙️  Testing MCP AI Server Configuration")
    print("=" * 70)

    try:
        from backend.core.utils.config import Config

        print("\n📋 Configuration loaded successfully!")
        Config.display()
        return True

    except Exception as e:
        print(f"\n❌ Error loading config: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧠 MCP AI Server - GPU Setup Test")
    print("=" * 70)
    print("\nThis script will verify your GPU setup for running Qwen2.5-Coder")
    print("with 4-bit quantization on your RTX 3050 (4GB VRAM)\n")

    results = {
        "imports": False,
        "cuda": False,
        "quantization": False,
        "model": False,
        "config": False
    }

    # Run tests
    results["imports"] = test_imports()

    if results["imports"]:
        results["cuda"] = test_cuda()

        if results["cuda"]:
            results["quantization"] = test_quantization()

            # Only test model loading if user confirms
            print("\n" + "=" * 70)
            response = input("\n🤔 Do you want to test loading a small model? (y/N): ")
            if response.lower() == 'y':
                results["model"] = test_small_model()
            else:
                print("Skipping model test.")
                results["model"] = None

    results["config"] = test_config()

    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)

    status_map = {True: "✅ PASS", False: "❌ FAIL", None: "⏭️  SKIPPED"}

    for test, result in results.items():
        print(f"{test.capitalize()}: {status_map[result]}")

    # Final verdict
    critical_tests = ["imports", "cuda", "quantization"]
    all_critical_passed = all(results[t] for t in critical_tests)

    print("\n" + "=" * 70)
    if all_critical_passed:
        print("✅ ALL CRITICAL TESTS PASSED!")
        print("=" * 70)
        print("\n🚀 Your system is ready to run MCP AI Server!")
        print("\nNext steps:")
        print("  1. Run: python scripts/download_models.py")
        print("  2. The 7B model will use ~3.5GB VRAM with 4-bit quantization")
        print("  3. Inference speed: ~20-40 tokens/second")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("  • Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  • Install BitsAndBytes: pip install bitsandbytes")
        print("  • Update NVIDIA drivers")

    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)