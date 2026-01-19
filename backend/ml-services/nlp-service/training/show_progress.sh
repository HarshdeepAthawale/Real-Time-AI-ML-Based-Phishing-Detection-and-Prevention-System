#!/bin/bash
# Quick progress display script

cd "$(dirname "$0")/.."

echo "=========================================="
echo "TRAINING PROGRESS - $(date '+%H:%M:%S')"
echo "=========================================="
echo ""

# Phishing Model
echo "📧 PHISHING MODEL:"
if [ -f "models/phishing-bert-v1/model.safetensors" ]; then
    SIZE=$(du -h models/phishing-bert-v1/model.safetensors 2>/dev/null | cut -f1)
    echo "  ✓ COMPLETED - Model saved ($SIZE)"
else
    echo "  Status: Not found"
fi
echo ""

# AI Detector
echo "🤖 AI DETECTOR:"
if [ -f "models/ai-detector-v1/model.safetensors" ]; then
    SIZE=$(du -h models/ai-detector-v1/model.safetensors 2>/dev/null | cut -f1)
    echo "  ✓ COMPLETED - Model saved ($SIZE)"
else
    # Check if process is running
    if pgrep -f "train_ai_detector" > /dev/null; then
        echo "  ⏳ Training in progress..."
        # Try to extract progress from log
        if [ -f "/tmp/ai_train_final.log" ]; then
            PROGRESS=$(tail -100 /tmp/ai_train_final.log 2>/dev/null | grep -oP '\d+%\s*\|\s*\d+/\d+' | tail -1)
            if [ -n "$PROGRESS" ]; then
                echo "  Progress: $PROGRESS"
            fi
            EPOCH=$(tail -100 /tmp/ai_train_final.log 2>/dev/null | grep -oP "'epoch':\s*[\d.]+" | tail -1 | grep -oP '[\d.]+')
            if [ -n "$EPOCH" ]; then
                echo "  Epoch: $EPOCH/3.0"
            fi
        fi
    else
        echo "  Status: Not running"
    fi
fi
echo ""
