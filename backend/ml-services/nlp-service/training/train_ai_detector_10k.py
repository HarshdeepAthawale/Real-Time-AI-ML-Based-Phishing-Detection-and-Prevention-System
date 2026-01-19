#!/usr/bin/env python3
"""
Automated training pipeline for AI Detector with 10k sample limit
This script automates the training process for faster hackathon/demo setup
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run automated training pipeline"""
    # Get script directory
    script_dir = Path(__file__).parent
    service_dir = script_dir.parent
    
    # Change to service directory
    os.chdir(service_dir)
    
    print("=" * 60)
    print("AI DETECTOR TRAINING PIPELINE (10k Samples)")
    print("=" * 60)
    print()
    print("This will train the AI detector model with:")
    print("  - Dataset: 10,000 samples (stratified)")
    print("  - Epochs: 3")
    print("  - Batch size: 16")
    print("  - Estimated time: ~11 minutes")
    print()
    
    # Check if virtual environment exists
    venv_path = service_dir / "venv"
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Installing dependencies...")
        pip_path = venv_path / "bin" / "pip"
        if not pip_path.exists():
            pip_path = venv_path / "Scripts" / "pip.exe"  # Windows
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    else:
        print("✓ Virtual environment found")
    
    # Determine Python executable in venv
    python_exe = venv_path / "bin" / "python"
    if not python_exe.exists():
        python_exe = venv_path / "Scripts" / "python.exe"  # Windows
    
    if not python_exe.exists():
        print("❌ Python executable not found in venv")
        sys.exit(1)
    
    # Check dependencies
    print("Checking dependencies...")
    try:
        subprocess.run(
            [str(python_exe), "-c", "import transformers; import torch; import datasets; import pandas"],
            check=True,
            capture_output=True
        )
        print("✓ All dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Missing dependencies. Installing...")
        pip_exe = venv_path / "bin" / "pip"
        if not pip_exe.exists():
            pip_exe = venv_path / "Scripts" / "pip.exe"
        subprocess.run([str(pip_exe), "install", "-r", "requirements.txt"], check=True)
    
    print()
    print("Starting training...")
    print("=" * 60)
    print()
    
    # Run training
    training_script = service_dir / "training" / "train_ai_detector.py"
    cmd = [
        str(python_exe),
        str(training_script),
        "--huggingface",
        "--max-samples", "10000",
        "--epochs", "3",
        "--batch-size", "16",
        "--lr", "2e-5",
        "--model", "roberta-base",
        "--output", "./models/ai-detector-v1"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("Training completed!")
    print("=" * 60)
    print()
    
    # Check if model was saved
    model_path = service_dir / "models" / "ai-detector-v1" / "model.safetensors"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model saved successfully ({size_mb:.1f} MB)")
        print()
        print("You can now use this model in the NLP service!")
    else:
        print("⚠ Warning: Model file not found. Training may have failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
