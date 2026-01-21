#!/usr/bin/env python3
"""
Model Verification Script

Checks if trained models exist and verifies their integrity.
If models don't exist, provides instructions for training.
"""

import os
import sys
from pathlib import Path

def check_model_exists(model_path: str, model_name: str) -> bool:
    """Check if a model exists and has required files"""
    if not model_path or not os.path.exists(model_path):
        return False
    
    # Check for required model files
    required_files = [
        "config.json",
        "pytorch_model.bin",  # or model.safetensors
        "tokenizer_config.json",
        "vocab.json"  # or vocab.txt depending on tokenizer
    ]
    
    found_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            found_files.append(file)
    
    # At minimum, we need config.json and a model file
    has_config = "config.json" in found_files
    has_model = any("pytorch_model" in f or "model.safetensors" in f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f)))
    
    if has_config and has_model:
        print(f"✅ {model_name}: Found at {model_path}")
        print(f"   Found files: {', '.join(found_files)}")
        return True
    else:
        print(f"⚠️  {model_name}: Incomplete at {model_path}")
        print(f"   Missing: config.json={not has_config}, model file={not has_model}")
        return False

def main():
    """Main verification function"""
    print("=" * 60)
    print("NLP Service Model Verification")
    print("=" * 60)
    print()
    
    model_dir = os.getenv("MODEL_DIR", "./models")
    model_dir = Path(model_dir).resolve()
    
    print(f"Model directory: {model_dir}")
    print()
    
    # Check phishing model
    phishing_paths = [
        model_dir / "phishing-bert-v1",
        model_dir / "phishing-roberta-v1",
    ]
    
    phishing_found = False
    for path in phishing_paths:
        if check_model_exists(str(path), "Phishing Classifier"):
            phishing_found = True
            break
    
    if not phishing_found:
        print("❌ Phishing Classifier: Not found")
        print("   Using pre-trained RoBERTa-base model (fallback)")
        print("   To train: python training/train_phishing_model.py")
    
    print()
    
    # Check AI detector model
    ai_paths = [
        model_dir / "ai-detector-v1",
        model_dir / "ai-detector-roberta-v1",
    ]
    
    ai_found = False
    for path in ai_paths:
        if check_model_exists(str(path), "AI Detector"):
            ai_found = True
            break
    
    if not ai_found:
        print("❌ AI Detector: Not found")
        print("   Using pre-trained RoBERTa-base model (fallback)")
        print("   To train: python training/train_ai_detector.py")
    
    print()
    print("=" * 60)
    
    if phishing_found and ai_found:
        print("✅ All models found and ready!")
        return 0
    elif phishing_found or ai_found:
        print("⚠️  Some models found. Service will use pre-trained fallbacks for missing models.")
        return 1
    else:
        print("ℹ️  No custom models found. Service will use pre-trained RoBERTa-base models.")
        print("   This is acceptable for development/testing.")
        print("   For production, train custom models using the training scripts.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
