#!/bin/bash
# Quick training script for AI Detector with 1k samples (Hackathon Fast Mode)
# Training time: ~2-3 minutes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SERVICE_DIR"

echo "=========================================="
echo "AI DETECTOR TRAINING - 1K SAMPLES (FAST)"
echo "=========================================="
echo ""
echo "Training configuration:"
echo "  - Dataset: 1,000 samples (stratified)"
echo "  - Epochs: 3"
echo "  - Batch size: 8"
echo "  - Estimated time: ~2-3 minutes"
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup_training_env.sh first"
    exit 1
fi

source venv/bin/activate

echo "Starting training..."
echo "=========================================="
echo ""

# Run training with 1k sample limit
python3 training/train_ai_detector.py \
    --huggingface \
    --max-samples 1000 \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-5 \
    --model roberta-base \
    --output ./models/ai-detector-v1

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""

# Check if model was saved
if [ -f "models/ai-detector-v1/model.safetensors" ]; then
    MODEL_SIZE=$(du -h models/ai-detector-v1/model.safetensors | cut -f1)
    echo "✓ Model saved successfully ($MODEL_SIZE)"
    echo ""
    echo "Model ready for hackathon demo!"
else
    echo "⚠ Warning: Model file not found."
    exit 1
fi
