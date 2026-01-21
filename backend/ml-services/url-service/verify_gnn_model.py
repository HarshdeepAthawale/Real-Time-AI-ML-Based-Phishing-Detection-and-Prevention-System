#!/usr/bin/env python3
"""
GNN Model Verification Script

Checks if trained GNN model exists and verifies its integrity.
If model doesn't exist, provides instructions for training.
"""

import os
import sys
from pathlib import Path

def check_model_exists(model_path: str) -> bool:
    """Check if a GNN model exists"""
    if not model_path or not os.path.exists(model_path):
        return False
    
    # Check if it's a file (PyTorch .pt file)
    if os.path.isfile(model_path):
        file_size = os.path.getsize(model_path)
        if file_size > 0:
            print(f"✅ GNN Model: Found at {model_path}")
            print(f"   File size: {file_size / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"⚠️  GNN Model: Empty file at {model_path}")
            return False
    
    # Check if it's a directory with model files
    if os.path.isdir(model_path):
        required_files = ["model.pt"]
        found_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
        if found_files:
            print(f"✅ GNN Model: Found in directory {model_path}")
            print(f"   Model files: {', '.join(found_files)}")
            return True
        else:
            print(f"⚠️  GNN Model: Directory exists but no .pt files found")
            return False
    
    return False

def main():
    """Main verification function"""
    print("=" * 60)
    print("URL Service GNN Model Verification")
    print("=" * 60)
    print()
    
    # Check multiple possible model paths
    model_paths = [
        os.getenv("GNN_MODEL_PATH", "./models/gnn-domain-classifier-v1/model.pt"),
        "./models/gnn-domain-classifier-v1/model.pt",
        "./models/gnn-domain-classifier-v1",
    ]
    
    model_found = False
    for path in model_paths:
        if path and check_model_exists(path):
            model_found = True
            break
    
    if not model_found:
        print("❌ GNN Model: Not found")
        print("   Using untrained model (random weights)")
        print("   To train: python training/train_gnn_model.py")
        print()
        print("   Note: Untrained model will provide random predictions.")
        print("   For production use, train the model on domain relationship data.")
    
    print()
    print("=" * 60)
    
    if model_found:
        print("✅ GNN model found and ready!")
        return 0
    else:
        print("ℹ️  No trained GNN model found. Service will use untrained model.")
        print("   This is acceptable for development/testing.")
        print("   For production, train the model using the training script.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
