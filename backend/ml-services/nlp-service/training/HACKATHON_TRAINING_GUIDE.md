# Hackathon Training Guide - Quick Setup

## Sample Size Comparison

### 1,000 Samples (Ultra Fast - ~2-3 minutes)
**Best for:** Quick demos, tight time constraints, proof of concept
- Training: ~700 samples
- Validation: ~150 samples  
- Test: ~150 samples
- Training time: **~2-3 minutes**
- Expected accuracy: **75-85%** (acceptable for demo)
- Risk: May overfit, less reliable

### 10,000 Samples (Recommended - ~15-20 minutes)
**Best for:** Better demo quality, more impressive results
- Training: ~7,000 samples
- Validation: ~1,500 samples
- Test: ~1,500 samples
- Training time: **~15-20 minutes**
- Expected accuracy: **85-92%** (good for demo)
- Risk: Low, well-balanced

### 100,000 Samples (Full Dataset - ~2 hours)
**Best for:** Production quality, best accuracy
- Training: ~70,000 samples
- Training time: **~2 hours**
- Expected accuracy: **90-95%** (production quality)
- Risk: Very low, best performance

## Quick Commands

### Train with 1k samples (Fastest)
```bash
cd backend/ml-services/nlp-service
source venv/bin/activate
python training/train_ai_detector.py \
    --huggingface \
    --max-samples 1000 \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-5 \
    --model roberta-base \
    --output ./models/ai-detector-v1
```

### Train with 10k samples (Recommended)
```bash
cd backend/ml-services/nlp-service
source venv/bin/activate
python training/train_ai_detector.py \
    --huggingface \
    --max-samples 10000 \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-5 \
    --model roberta-base \
    --output ./models/ai-detector-v1
```

## Recommendations for Hackathons

### If you have < 30 minutes:
✅ **Use 1,000 samples** - Fast enough to train during the hackathon, good enough for demo

### If you have 30-60 minutes:
✅ **Use 10,000 samples** - Better quality, still fast enough

### If you have time before hackathon:
✅ **Pre-train with 100k samples** - Best quality, train beforehand

## Tips

1. **Pre-train if possible**: Train the model before the hackathon starts
2. **Use 1k for quick iterations**: If you need to retrain multiple times
3. **Use 10k for final demo**: Better accuracy impresses judges
4. **Monitor training**: Use `tail -f /tmp/ai_train_10k.log` to watch progress

## Expected Results

| Samples | Training Time | Accuracy | F1 Score | Demo Quality |
|---------|--------------|----------|----------|--------------|
| 1k      | 2-3 min      | 75-85%   | 0.75-0.85| ⭐⭐⭐ Good |
| 10k     | 15-20 min    | 85-92%   | 0.85-0.92| ⭐⭐⭐⭐ Great |
| 100k    | ~2 hours     | 90-95%   | 0.90-0.95| ⭐⭐⭐⭐⭐ Excellent |

## Quick Decision Guide

- **Need it NOW?** → 1k samples
- **Have 20 minutes?** → 10k samples (recommended)
- **Want best quality?** → 100k samples (pre-train)
