# Quick Start Guide - Training Models

This is a step-by-step guide to download datasets and train your models.

## 🚀 Quick Start (5 minutes)

### Step 1: Install Training Dependencies

```bash
cd backend/ml-services/nlp-service
pip install datasets pandas scikit-learn
```

### Step 2: Download AI Detection Dataset (Automatic)

```bash
python training/download_datasets.py --ai
```

This will automatically download the AI detection dataset from Hugging Face and save it to `data/raw/ai_detection.csv`.

### Step 3: Download Phishing Email Dataset (Manual)

The phishing dataset needs to be downloaded manually. Here are the easiest options:

#### Option A: UTwente Dataset (Easiest - 2,000 emails)
1. Visit: https://research.utwente.nl/en/datasets/phishing-validation-emails-dataset/
2. Download the CSV file
3. Save as: `data/raw/phishing_emails.csv`
4. Ensure it has columns: `text` and `label` (0=legitimate, 1=phishing)

#### Option B: Sting9 Dataset (Larger - Recommended)
1. Visit: https://sting9.org/dataset
2. Register (free)
3. Download SQL dump or use API
4. Convert to CSV with `text` and `label` columns
5. Save as: `data/raw/phishing_emails.csv`

See `DATASET_GUIDE.md` for more options and conversion scripts.

### Step 4: Prepare Data Directories

```bash
mkdir -p data/{raw,processed}
mkdir -p models/{phishing-bert-v1,ai-detector-v1}
```

### Step 5: Train AI Detection Model

```bash
# Using Hugging Face dataset (easiest)
python training/train_ai_detector.py --huggingface

# Or using downloaded CSV
python training/train_ai_detector.py --dataset data/raw/ai_detection.csv
```

### Step 6: Train Phishing Detection Model

```bash
python training/train_phishing_model.py --dataset data/raw/phishing_emails.csv
```

## 📋 Complete Example Workflow

```bash
# 1. Navigate to service directory
cd backend/ml-services/nlp-service

# 2. Install dependencies
pip install datasets pandas scikit-learn

# 3. Create directories
mkdir -p data/{raw,processed} models/{phishing-bert-v1,ai-detector-v1}

# 4. Download AI detection dataset
python training/download_datasets.py --ai

# 5. Download phishing dataset info
python training/download_datasets.py --phishing-info

# 6. (Manual) Download phishing dataset from one of the sources
# Save as data/raw/phishing_emails.csv

# 7. Train AI detector (takes ~30-60 minutes on CPU)
python training/train_ai_detector.py --huggingface --epochs 3

# 8. Train phishing detector (takes ~1-2 hours on CPU)
python training/train_phishing_model.py --dataset data/raw/phishing_emails.csv --epochs 3

# 9. Test the service
python -m src.main
# Visit http://localhost:8000/docs to test the API
```

## ⚙️ Training Options

### Adjust Training Parameters

```bash
# Train with custom parameters
python training/train_phishing_model.py \
    --dataset data/raw/phishing_emails.csv \
    --epochs 5 \
    --batch-size 32 \
    --lr 3e-5 \
    --model distilbert-base-uncased

# Train AI detector with RoBERTa
python training/train_ai_detector.py \
    --huggingface \
    --model roberta-base \
    --epochs 5 \
    --batch-size 16
```

### GPU Training

If you have a GPU:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Training will automatically use GPU if available
python training/train_phishing_model.py --dataset data/raw/phishing_emails.csv
```

## 📊 Expected Results

After training, you should see:

1. **Model files** in:
   - `models/phishing-bert-v1/` (config.json, pytorch_model.bin, tokenizer files)
   - `models/ai-detector-v1/` (config.json, pytorch_model.bin, tokenizer files)

2. **Training logs** in:
   - `models/phishing-bert-v1/logs/`
   - `models/ai-detector-v1/logs/`

3. **Validation accuracy** should be:
   - Phishing detection: >85% (with good dataset)
   - AI detection: >90% (with Hugging Face dataset)

## 🧪 Test Your Models

After training, test the service:

```bash
# Start the service
python -m src.main

# In another terminal, test the API
curl -X POST "http://localhost:8000/api/v1/analyze-text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Urgent! Verify your account immediately or it will be suspended.",
    "include_features": false
  }'
```

## 🐛 Troubleshooting

### "Dataset not found"
- Check file path: `ls data/raw/`
- Ensure CSV has `text` and `label` columns

### "Out of memory"
- Reduce batch size: `--batch-size 8`
- Use smaller model: `--model distilbert-base-uncased`

### "CUDA out of memory" (GPU)
- Reduce batch size
- Use gradient accumulation (modify training script)

### Low accuracy
- Increase training epochs: `--epochs 5`
- Check dataset quality and balance
- Try different learning rate: `--lr 3e-5`

## 📚 Next Steps

- Read `DATASET_GUIDE.md` for detailed dataset information
- Read `README.md` for service documentation
- Check training logs for detailed metrics
- Fine-tune hyperparameters based on validation results

## 💡 Tips

1. **Start small**: Use UTwente dataset (2K samples) for quick testing
2. **Use GPU**: Training is 10x faster on GPU
3. **Monitor training**: Check logs in `models/*/logs/`
4. **Save checkpoints**: Models are saved during training
5. **Validate early**: Check validation accuracy during training
