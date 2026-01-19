#!/bin/bash
# Automated training pipeline for AI Detector with 10k sample limit
# This script automates the training process for faster hackathon/demo setup

set -e  # Exit on error

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SERVICE_DIR"

echo "=========================================="
echo "AI DETECTOR TRAINING PIPELINE (10k Samples)"
echo "=========================================="
echo ""
echo "This will train the AI detector model with:"
echo "  - Dataset: 10,000 samples (stratified)"
echo "  - Epochs: 3"
echo "  - Batch size: 16"
echo "  - Estimated time: ~11 minutes"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✓ Virtual environment found"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import transformers; import torch; import datasets; import pandas; print('✓ All dependencies installed')" || {
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
}

echo ""
echo "Starting training..."
echo "=========================================="
echo ""

# Run training with 10k sample limit
python3 training/train_ai_detector.py \
    --huggingface \
    --max-samples 10000 \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5 \
    --model roberta-base \
    --output ./models/ai-detector-v1

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Model saved to: ./models/ai-detector-v1/"
echo ""

# Check if model was saved successfully
if [ -f "models/ai-detector-v1/model.safetensors" ]; then
    MODEL_SIZE=$(du -h models/ai-detector-v1/model.safetensors | cut -f1)
    echo "✓ Model saved successfully ($MODEL_SIZE)"
    echo ""
    echo "You can now use this model in the NLP service!"
else
    echo "⚠ Warning: Model file not found. Training may have failed."
    exit 1
fi
