#!/usr/bin/env python3
"""
Model Download Script for MCP AI Server
Downloads Qwen2.5-Coder and embedding models locally
"""
import sys
import os
from pathlib import Path

# Add parent directory to path to import config
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.config import settings as Config


def download_llm_model():
    """Download the primary LLM model"""
    print("=" * 70)
    print(f"📥 Downloading {Config.MODEL_NAME}")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_name = Config.MODEL_NAME
        save_dir = Config.MODEL_DOWNLOAD_DIR / model_name.replace("/", "--")

        print(f"\n📍 Model: {model_name}")
        print(f"💾 Save Location: {save_dir}")
        print(f"🔧 Device: {Config.get_device()}")

        # Show GPU info
        gpu_info = Config.get_gpu_info()
        if gpu_info.get("available") and "name" in gpu_info:
            print(f"🎮 GPU: {gpu_info['name']}")
            print(f"   Total VRAM: {gpu_info['memory_total']:.2f} GB")

        # Estimate model size based on name
        if "0.5B" in model_name or "500M" in model_name:
            size_info = "~1GB"
            vram_info = "~0.5GB VRAM"
        elif "1.5B" in model_name:
            size_info = "~3GB"
            vram_info = "~1.5GB VRAM"
        elif "3B" in model_name:
            size_info = "~6GB"
            vram_info = "~3GB VRAM"
        elif "7B" in model_name:
            size_info = "~14GB"
            vram_info = "~3.5GB VRAM (with 4-bit)"
        else:
            size_info = "Unknown"
            vram_info = "Unknown"

        print(f"📊 Estimated download: {size_info}")
        print(f"💾 Estimated VRAM usage: {vram_info}")

        # Check if already downloaded
        if save_dir.exists() and (save_dir / "config.json").exists():
            print(f"\n✅ Model already exists at {save_dir}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                print("Skipping download.")
                return str(save_dir)

        print("\n⏳ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=Config.MODEL_TRUST_REMOTE_CODE
        )

        print(f"⏳ Downloading model ({size_info})...")
        if "0.5B" in model_name or "500M" in model_name:
            print("   ⚡ This should be quick (2-5 minutes)!")
        else:
            print("   Tip: Grab a coffee ☕, this might take 10-30 minutes")

        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None,  # Don't load to device yet
            trust_remote_code=Config.MODEL_TRUST_REMOTE_CODE,
            low_cpu_mem_usage=True
        )

        print(f"\n💾 Saving model to {save_dir}...")
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Clear memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n✅ Model successfully downloaded to: {save_dir}")
        print(f"📊 Download size: {size_info}")
        print(f"💾 Runtime VRAM usage: {vram_info}")

        return str(save_dir)

    except ImportError as e:
        print(f"\n❌ Error: Missing required package: {e}")
        print("Please install: pip install transformers torch accelerate")
        return None
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_embedding_model():
    """Download the embedding model for RAG"""
    print("\n" + "=" * 70)
    print("📥 Downloading Embedding Model")
    print("=" * 70)

    try:
        from sentence_transformers import SentenceTransformer

        model_name = Config.EMBEDDING_MODEL
        save_dir = Config.EMBEDDING_DOWNLOAD_DIR / model_name.replace("/", "--")

        print(f"\n📍 Model: {model_name}")
        print(f"💾 Save Location: {save_dir}")

        # Check if already downloaded
        if save_dir.exists() and list(save_dir.glob("*.bin")):
            print(f"\n✅ Embedding model already exists at {save_dir}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                print("Skipping download.")
                return str(save_dir)

        print("\n⏳ Downloading embedding model (this is quick, ~80MB)...")

        model = SentenceTransformer(model_name)

        print(f"\n💾 Saving embedding model to {save_dir}...")
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(save_dir))

        print(f"\n✅ Embedding model successfully downloaded to: {save_dir}")

        return str(save_dir)

    except ImportError as e:
        print(f"\n❌ Error: Missing required package: {e}")
        print("Please install: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"\n❌ Error downloading embedding model: {e}")
        return None


def verify_downloads():
    """Verify that models were downloaded correctly"""
    print("\n" + "=" * 70)
    print("🔍 Verifying Downloads")
    print("=" * 70)

    llm_path = Config.MODEL_DOWNLOAD_DIR / Config.MODEL_NAME.replace("/", "--")
    embed_path = Config.EMBEDDING_DOWNLOAD_DIR / Config.EMBEDDING_MODEL.replace("/", "--")

    llm_ok = llm_path.exists() and (llm_path / "config.json").exists()
    embed_ok = embed_path.exists() and list(embed_path.glob("*.bin"))

    print(f"\n📦 LLM Model: {'✅ OK' if llm_ok else '❌ Not Found'}")
    if llm_ok:
        print(f"   Path: {llm_path}")

    print(f"\n📦 Embedding Model: {'✅ OK' if embed_ok else '❌ Not Found'}")
    if embed_ok:
        print(f"   Path: {embed_path}")

    if llm_ok and embed_ok:
        print("\n" + "=" * 70)
        print("✅ All models downloaded successfully!")
        print("=" * 70)
        print("\n📝 Next steps:")
        print("1. Copy .env.example to .env")
        print("2. Set MODEL_USE_LOCAL=True in your .env")
        print("3. Run: python demo.py")
        return True
    else:
        print("\n⚠️  Some models are missing. Please run the download again.")
        return False


def main():
    """Main download orchestration"""
    print("\n🤖 MCP AI Server - Model Download Tool")
    print("=" * 70)

    # Ensure directories exist
    Config.ensure_directories()

    model_name = Config.MODEL_NAME

    # Determine size based on model name
    if "0.5B" in model_name or "500M" in model_name:
        size_gb = 1
        vram_gb = 0.5
    elif "1.5B" in model_name:
        size_gb = 3
        vram_gb = 1.5
    elif "3B" in model_name:
        size_gb = 6
        vram_gb = 3
    elif "7B" in model_name:
        size_gb = 14
        vram_gb = 3.5
    else:
        size_gb = "?"
        vram_gb = "?"

    print(f"\nModel to download: {model_name}")
    print("\nThis script will download:")
    print(f"  1. {model_name} (~{size_gb}GB)")
    print(f"     Runtime VRAM: ~{vram_gb}GB")
    print(f"  2. all-MiniLM-L6-v2 embeddings (~80MB)")
    print(f"\nTotal download size: ~{size_gb + 0.08:.1f}GB")
    print(f"Save location: {Config.MODELS_CACHE_DIR}")

    response = input("\nDo you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    # Download LLM
    llm_path = download_llm_model()

    # Download Embedding Model
    embed_path = download_embedding_model()

    # Verify
    verify_downloads()

    print("\n✨ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


def download_llm_model():
    """Download the primary LLM model"""
    print("=" * 70)
    print(f"📥 Downloading {Config.MODEL_NAME}")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_name = Config.MODEL_NAME
        save_dir = Config.MODEL_DOWNLOAD_DIR / model_name.replace("/", "--")

        print(f"\n📍 Model: {model_name}")
        print(f"💾 Save Location: {save_dir}")
        print(f"🔧 Device: {Config.get_device()}")

        # Show GPU info
        gpu_info = Config.get_gpu_info()
        if gpu_info.get("available") and "name" in gpu_info:
            print(f"🎮 GPU: {gpu_info['name']}")
            print(f"   Total VRAM: {gpu_info['memory_total']:.2f} GB")

        # Estimate model size based on name
        if "0.5B" in model_name or "500M" in model_name:
            size_info = "~1GB"
            vram_info = "~0.5GB VRAM"
        elif "1.5B" in model_name:
            size_info = "~3GB"
            vram_info = "~1.5GB VRAM"
        elif "3B" in model_name:
            size_info = "~6GB"
            vram_info = "~3GB VRAM"
        elif "7B" in model_name:
            size_info = "~14GB"
            vram_info = "~3.5GB VRAM (with 4-bit)"
        else:
            size_info = "Unknown"
            vram_info = "Unknown"

        print(f"📊 Estimated download: {size_info}")
        print(f"💾 Estimated VRAM usage: {vram_info}")

        # Check if already downloaded
        if save_dir.exists() and (save_dir / "config.json").exists():
            print(f"\n✅ Model already exists at {save_dir}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                print("Skipping download.")
                return str(save_dir)

        print("\n⏳ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=Config.MODEL_TRUST_REMOTE_CODE
        )

        print(f"⏳ Downloading model ({size_info})...")
        if "0.5B" in model_name or "500M" in model_name:
            print("   ⚡ This should be quick (2-5 minutes)!")
        else:
            print("   Tip: Grab a coffee ☕, this might take 10-30 minutes")

        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None,  # Don't load to device yet
            trust_remote_code=Config.MODEL_TRUST_REMOTE_CODE,
            low_cpu_mem_usage=True
        )

        print(f"\n💾 Saving model to {save_dir}...")
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Clear memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n✅ Model successfully downloaded to: {save_dir}")
        print(f"📊 Download size: {size_info}")
        print(f"💾 Runtime VRAM usage: {vram_info}")

        return str(save_dir)

    except ImportError as e:
        print(f"\n❌ Error: Missing required package: {e}")
        print("Please install: pip install transformers torch accelerate")
        return None
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_embedding_model():
    """Download the embedding model for RAG"""
    print("\n" + "=" * 70)
    print("📥 Downloading Embedding Model")
    print("=" * 70)

    try:
        from sentence_transformers import SentenceTransformer

        model_name = Config.EMBEDDING_MODEL
        save_dir = Config.EMBEDDING_DOWNLOAD_DIR / model_name.replace("/", "--")

        print(f"\n📍 Model: {model_name}")
        print(f"💾 Save Location: {save_dir}")

        # Check if already downloaded
        if save_dir.exists() and list(save_dir.glob("*.bin")):
            print(f"\n✅ Embedding model already exists at {save_dir}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                print("Skipping download.")
                return str(save_dir)

        print("\n⏳ Downloading embedding model (this is quick, ~80MB)...")

        model = SentenceTransformer(model_name)

        print(f"\n💾 Saving embedding model to {save_dir}...")
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(save_dir))

        print(f"\n✅ Embedding model successfully downloaded to: {save_dir}")

        return str(save_dir)

    except ImportError as e:
        print(f"\n❌ Error: Missing required package: {e}")
        print("Please install: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"\n❌ Error downloading embedding model: {e}")
        return None


def verify_downloads():
    """Verify that models were downloaded correctly"""
    print("\n" + "=" * 70)
    print("🔍 Verifying Downloads")
    print("=" * 70)

    llm_path = Config.MODEL_DOWNLOAD_DIR / Config.MODEL_NAME.replace("/", "--")
    embed_path = Config.EMBEDDING_DOWNLOAD_DIR / Config.EMBEDDING_MODEL.replace("/", "--")

    llm_ok = llm_path.exists() and (llm_path / "config.json").exists()
    embed_ok = embed_path.exists() and list(embed_path.glob("*.bin"))

    print(f"\n📦 LLM Model: {'✅ OK' if llm_ok else '❌ Not Found'}")
    if llm_ok:
        print(f"   Path: {llm_path}")

    print(f"\n📦 Embedding Model: {'✅ OK' if embed_ok else '❌ Not Found'}")
    if embed_ok:
        print(f"   Path: {embed_path}")

    if llm_ok and embed_ok:
        print("\n" + "=" * 70)
        print("✅ All models downloaded successfully!")
        print("=" * 70)
        print("\n📝 Next steps:")
        print("1. Copy .env.example to .env")
        print("2. Set MODEL_USE_LOCAL=True in your .env")
        print("3. Run: python demo.py")
        return True
    else:
        print("\n⚠️  Some models are missing. Please run the download again.")
        return False


def main():
    """Main download orchestration"""
    print("\n🤖 MCP AI Server - Model Download Tool")
    print("=" * 70)

    # Ensure directories exist
    Config.ensure_directories()

    model_name = Config.MODEL_NAME

    # Determine size based on model name
    if "0.5B" in model_name or "500M" in model_name:
        size_gb = 1
        vram_gb = 0.5
    elif "1.5B" in model_name:
        size_gb = 3
        vram_gb = 1.5
    elif "3B" in model_name:
        size_gb = 6
        vram_gb = 3
    elif "7B" in model_name:
        size_gb = 14
        vram_gb = 3.5
    else:
        size_gb = "?"
        vram_gb = "?"

    print(f"\nModel to download: {model_name}")
    print("\nThis script will download:")
    print(f"  1. {model_name} (~{size_gb}GB)")
    print(f"     Runtime VRAM: ~{vram_gb}GB")
    print(f"  2. all-MiniLM-L6-v2 embeddings (~80MB)")
    print(f"\nTotal download size: ~{size_gb + 0.08:.1f}GB")
    print(f"Save location: {Config.MODELS_CACHE_DIR}")

    response = input("\nDo you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    # Download LLM
    llm_path = download_llm_model()

    # Download Embedding Model
    embed_path = download_embedding_model()

    # Verify
    verify_downloads()

    print("\n✨ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)