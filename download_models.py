#!/usr/bin/env python3
"""
Model Download Script for Syllabus Checker

This script downloads all available embedding models to ensure they're available
offline. This is useful for:
1. Initial setup
2. Environments with unreliable internet
3. Pre-deployment preparation

Usage:
    python download_models.py [options]

Options:
    --model MODEL_KEY    Download specific model (e.g., bge-large-en)
    --all               Download all available models
    --list              List all available models
    --check             Check which models are already downloaded
"""

import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from embeddings import (
    AVAILABLE_MODELS, 
    ensure_model_downloaded, 
    preload_default_models,
    EmbeddingGenerator
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_available_models():
    """List all available embedding models."""
    print("\n📋 Available Embedding Models:")
    print("=" * 50)
    
    for key, config in AVAILABLE_MODELS.items():
        print(f"🔹 {key}")
        print(f"   Name: {config['model_name']}")
        print(f"   Description: {config['description']}")
        print(f"   Dimension: {config['dimension']}")
        print()


def check_downloaded_models():
    """Check which models are already downloaded."""
    print("\n🔍 Checking Downloaded Models:")
    print("=" * 50)
    
    models_dir = "embeddingmodels"
    
    for key in AVAILABLE_MODELS.keys():
        try:
            # Create a temporary generator to check if model exists
            model_path = Path(models_dir) / key
            
            if model_path.exists():
                # Try to validate the model
                generator = EmbeddingGenerator(model_key=key, models_dir=models_dir)
                print(f"✅ {key} - Downloaded and validated")
            else:
                print(f"❌ {key} - Not downloaded")
                
        except Exception as e:
            print(f"⚠️  {key} - Downloaded but corrupted ({str(e)})")
    print()


def download_specific_model(model_key: str):
    """Download a specific model."""
    if model_key not in AVAILABLE_MODELS:
        print(f"❌ Error: Model '{model_key}' not found.")
        print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return False
    
    print(f"\n📦 Downloading model: {model_key}")
    print("=" * 50)
    
    config = AVAILABLE_MODELS[model_key]
    print(f"Model Name: {config['model_name']}")
    print(f"Description: {config['description']}")
    print(f"This may take several minutes depending on your internet connection...")
    print()
    
    success = ensure_model_downloaded(model_key)
    
    if success:
        print(f"✅ Successfully downloaded: {model_key}")
    else:
        print(f"❌ Failed to download: {model_key}")
    
    return success


def download_all_models():
    """Download all available models."""
    print("\n📦 Downloading All Models:")
    print("=" * 50)
    print("This will download all available embedding models.")
    print("This may take 30+ minutes depending on your internet connection.")
    print()
    
    # Ask for confirmation
    response = input("Do you want to continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Download cancelled.")
        return False
    
    print("\nStarting download of all models...")
    results = preload_default_models()
    
    print("\n📊 Download Results:")
    print("=" * 30)
    
    successful = 0
    for model_key, success in results.items():
        if success:
            print(f"✅ {model_key}")
            successful += 1
        else:
            print(f"❌ {model_key}")
    
    print(f"\nCompleted: {successful}/{len(results)} models downloaded successfully")
    return successful == len(results)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download embedding models for Syllabus Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_models.py --list
    python download_models.py --check
    python download_models.py --model bge-large-en
    python download_models.py --all
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        help="Download specific model (e.g., bge-large-en)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Download all available models"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available models"
    )
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check which models are already downloaded"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🤖 Syllabus Checker - Model Download Tool")
    print("=" * 50)
    
    # Handle different commands
    if args.list:
        list_available_models()
    elif args.check:
        check_downloaded_models()
    elif args.model:
        download_specific_model(args.model)
    elif args.all:
        download_all_models()
    else:
        # No arguments provided, show help
        parser.print_help()
        print("\n💡 Tip: Start with --list to see available models")
        print("💡 Tip: Use --check to see what's already downloaded")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)