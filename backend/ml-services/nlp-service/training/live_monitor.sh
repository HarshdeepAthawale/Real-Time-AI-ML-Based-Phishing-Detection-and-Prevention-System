#!/bin/bash
# Live training progress monitor

cd "$(dirname "$0")/.."

echo "=========================================="
echo "LIVE TRAINING PROGRESS MONITOR"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "TRAINING PROGRESS - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Phishing Model
    echo "📧 PHISHING MODEL:"
    if [ -f "/tmp/phishing_train_final.log" ]; then
        PHISHING_PROGRESS=$(tail -100 /tmp/phishing_train_final.log 2>/dev/null | grep -oP '\d+%' | tail -1)
        PHISHING_STEPS=$(tail -100 /tmp/phishing_train_final.log 2>/dev/null | grep -oP '\d+/\d+\s+\[.*\]' | tail -1)
        PHISHING_EPOCH=$(tail -100 /tmp/phishing_train_final.log 2>/dev/null | grep -oP "'epoch':\s*[\d.]+" | tail -1 | grep -oP '[\d.]+')
        PHISHING_ACC=$(tail -100 /tmp/phishing_train_final.log 2>/dev/null | grep -oP "'eval_accuracy':\s*[\d.]+" | tail -1 | grep -oP '[\d.]+')
        
        if [ -n "$PHISHING_PROGRESS" ]; then
            echo "  Progress: $PHISHING_PROGRESS"
            [ -n "$PHISHING_STEPS" ] && echo "  Steps: $PHISHING_STEPS"
            [ -n "$PHISHING_EPOCH" ] && echo "  Epoch: $PHISHING_EPOCH/3.0"
            [ -n "$PHISHING_ACC" ] && echo "  Accuracy: $PHISHING_ACC"
        else
            if [ -f "models/phishing-bert-v1/model.safetensors" ]; then
                SIZE=$(du -h models/phishing-bert-v1/model.safetensors 2>/dev/null | cut -f1)
                echo "  ✓ Training COMPLETED - Model saved ($SIZE)"
            else
                echo "  Status: No active progress detected"
            fi
        fi
    else
        echo "  No log file found"
    fi
    echo ""
    
    # AI Detector
    echo "🤖 AI DETECTOR:"
    if [ -f "/tmp/ai_train_final.log" ]; then
        AI_PROGRESS=$(tail -100 /tmp/ai_train_final.log 2>/dev/null | grep -oP '\d+%' | tail -1)
        AI_STEPS=$(tail -100 /tmp/ai_train_final.log 2>/dev/null | grep -oP '\d+/\d+\s+\[.*\]' | tail -1)
        AI_EPOCH=$(tail -100 /tmp/ai_train_final.log 2>/dev/null | grep -oP "'epoch':\s*[\d.]+" | tail -1 | grep -oP '[\d.]+')
        AI_ACC=$(tail -100 /tmp/ai_train_final.log 2>/dev/null | grep -oP "'eval_accuracy':\s*[\d.]+" | tail -1 | grep -oP '[\d.]+')
        
        if [ -n "$AI_PROGRESS" ]; then
            echo "  Progress: $AI_PROGRESS"
            [ -n "$AI_STEPS" ] && echo "  Steps: $AI_STEPS"
            [ -n "$AI_EPOCH" ] && echo "  Epoch: $AI_EPOCH/3.0"
            [ -n "$AI_ACC" ] && echo "  Accuracy: $AI_ACC"
        else
            if [ -f "models/ai-detector-v1/model.safetensors" ]; then
                SIZE=$(du -h models/ai-detector-v1/model.safetensors 2>/dev/null | cut -f1)
                echo "  ✓ Training COMPLETED - Model saved ($SIZE)"
            else
                # Check if process is running
                if pgrep -f "train_ai_detector" > /dev/null; then
                    echo "  ⏳ Training in progress (process running)"
                else
                    echo "  Status: Training stopped or not started"
                fi
            fi
        fi
    else
        echo "  No log file found"
    fi
    echo ""
    
    echo "=========================================="
    echo "Refreshing in 5 seconds... (Ctrl+C to stop)"
    sleep 5
done
