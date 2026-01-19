# Dataset Download and Training Guide

This guide will help you download datasets and train the NLP models for phishing detection and AI content detection.

## 📥 Dataset Sources

### 1. Phishing Email Datasets

#### Option A: Sting9 Research Initiative (Recommended)
- **URL**: https://sting9.org/dataset
- **License**: CC0 (Public Domain)
- **Size**: Large, growing collection
- **Format**: SQL dump, REST API, or CSV
- **Features**: Full email bodies, headers, metadata, anonymized
- **Update Frequency**: Weekly

**Download Steps:**
1. Visit https://sting9.org/dataset
2. Register/Login (free)
3. Download SQL dump or use REST API
4. Convert to CSV format (see conversion script below)

#### Option B: Seven Phishing Email Datasets (Figshare)
- **URL**: https://figshare.com/articles/dataset/Curated_Dataset_-_Phishing_Email/24899952
- **Size**: ~203,000 emails
- **Format**: CSV
- **Features**: Full body text, subject, metadata

**Download Steps:**
1. Visit the Figshare link
2. Click "Download" button
3. Extract the CSV file(s)
4. Ensure columns include: `text` (or `body`) and `label` (or `type`)

#### Option C: UTwente Phishing Validation Dataset
- **URL**: https://research.utwente.nl/en/datasets/phishing-validation-emails-dataset/
- **Size**: 2,000 emails
- **Format**: CSV (~200 KB)
- **License**: Check usage terms
- **Good for**: Quick testing/validation

#### Option D: EPVME Dataset (GitHub)
- **URL**: https://github.com/sunknighteric/EPVME-Dataset
- **Size**: 37,283 malicious emails
- **Format**: Various (check repository)
- **Features**: Headers + body, multiple attack types

### 2. AI-Generated Text Detection Datasets

#### Option A: Human vs AI Sentences (Hugging Face) - Recommended
- **URL**: https://huggingface.co/datasets/shahxeebhassan/human_vs_ai_sentences
- **Size**: 105,000 sentences
- **License**: MIT
- **Format**: Hugging Face Datasets (easy to load)

**Download Steps:**
```python
from datasets import load_dataset

dataset = load_dataset("shahxeebhassan/human_vs_ai_sentences")
# Automatically downloads and caches
```

#### Option B: Human vs LLM Text Corpus (Innovatiana)
- **URL**: https://www.innovatiana.com/en/datasets/human-vs-llm-text-corpus
- **Size**: ~788,000 text entries
- **License**: MIT
- **Format**: CSV
- **Features**: Various LLMs, ready-to-use

#### Option C: Comprehensive Dataset (Roy et al., 2025)
- **URL**: https://huggingface.co/datasets/gsingh1-py/train
- **Size**: 58,000+ samples
- **Format**: Hugging Face Datasets
- **Features**: NY Times articles + synthetic versions from multiple LLMs

## 🛠️ Dataset Preparation

### Step 1: Create Data Directory

```bash
mkdir -p data/raw
mkdir -p data/processed
```

### Step 2: Download Datasets

#### For Phishing Emails (using Sting9 as example):

1. **Download from website** or use their API
2. **Save to**: `data/raw/phishing_emails.csv`

Required CSV format:
```csv
text,label
"Your account has been suspended. Click here to verify.",1
"Meeting scheduled for tomorrow at 3 PM.",0
```

Where:
- `text`: Email content (subject + body combined)
- `label`: 0 = legitimate, 1 = phishing

#### For AI Detection (using Hugging Face):

The Hugging Face dataset can be loaded directly in Python (see training script).

Or download and save as CSV:
```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("shahxeebhassan/human_vs_ai_sentences")
df = pd.DataFrame(dataset['train'])
df.to_csv('data/raw/ai_detection.csv', index=False)
```

### Step 3: Prepare Datasets

Use the data preparation script:

```bash
cd backend/ml-services/nlp-service

# For phishing emails
python -c "
from training.data_preparation import prepare_phishing_dataset
prepare_phishing_dataset('data/raw/phishing_emails.csv', 'data/processed/phishing_dataset.json')
"

# For AI detection
python -c "
from training.data_preparation import prepare_ai_detection_dataset
prepare_ai_detection_dataset('data/raw/ai_detection.csv', 'data/processed/ai_dataset.json')
"
```

## 🚀 Training the Models

### Prerequisites

Install additional training dependencies:

```bash
pip install datasets pandas scikit-learn
```

### Train Phishing Detection Model

```bash
cd backend/ml-services/nlp-service
python training/train_phishing_model.py
```

The script will:
1. Load your dataset
2. Tokenize the data
3. Fine-tune DistilBERT
4. Save model to `models/phishing-bert-v1/`

### Train AI Detection Model

```bash
python training/train_ai_detector.py
```

The script will:
1. Load your dataset
2. Tokenize the data
3. Fine-tune RoBERTa
4. Save model to `models/ai-detector-v1/`

## 📝 Quick Start Example

### Complete Workflow

```bash
# 1. Create directories
mkdir -p data/{raw,processed}
mkdir -p models/{phishing-bert-v1,ai-detector-v1}

# 2. Download datasets (choose one method)

# Method A: Using Hugging Face (for AI detection)
python << EOF
from datasets import load_dataset
import pandas as pd

# Download AI detection dataset
ai_dataset = load_dataset("shahxeebhassan/human_vs_ai_sentences")
df_ai = pd.DataFrame(ai_dataset['train'])
df_ai.to_csv('data/raw/ai_detection.csv', index=False)
print(f"Downloaded {len(df_ai)} AI detection samples")
EOF

# Method B: Manual download for phishing emails
# Visit https://sting9.org/dataset and download
# Save as data/raw/phishing_emails.csv

# 3. Prepare datasets
python -c "
from training.data_preparation import prepare_phishing_dataset, prepare_ai_detection_dataset
prepare_phishing_dataset('data/raw/phishing_emails.csv', 'data/processed/phishing.json')
prepare_ai_detection_dataset('data/raw/ai_detection.csv', 'data/processed/ai.json')
"

# 4. Train models
python training/train_phishing_model.py
python training/train_ai_detector.py
```

## 🔧 Converting Different Dataset Formats

### Converting SQL to CSV

If you downloaded a SQL dump:

```python
import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect('phishing_emails.db')

# Query and convert to DataFrame
df = pd.read_sql_query("""
    SELECT 
        body as text,
        CASE WHEN type = 'phishing' THEN 1 ELSE 0 END as label
    FROM emails
""", conn)

# Save as CSV
df.to_csv('data/raw/phishing_emails.csv', index=False)
```

### Converting JSON to CSV

```python
import json
import pandas as pd

with open('phishing_emails.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv('data/raw/phishing_emails.csv', index=False)
```

### Extracting Email Body from Raw Emails

If you have raw email files:

```python
import email
import os
import pandas as pd
from training.data_preparation import EmailParser

parser = EmailParser()
emails = []

for filename in os.listdir('raw_emails/'):
    with open(f'raw_emails/{filename}', 'r') as f:
        raw_email = f.read()
        parsed = parser.parse(raw_email)
        emails.append({
            'text': f"{parsed['subject']} {parsed['body_text']}",
            'label': 1 if 'phishing' in filename else 0
        })

df = pd.DataFrame(emails)
df.to_csv('data/raw/phishing_emails.csv', index=False)
```

## ⚙️ Training Configuration

### Adjust Training Parameters

Edit `training/train_phishing_model.py`:

```python
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,              # Increase for better accuracy
    per_device_train_batch_size=16,  # Adjust based on GPU memory
    per_device_eval_batch_size=16,
    learning_rate=2e-5,              # Learning rate
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,     # Load best model
    metric_for_best_model="accuracy",
)
```

### GPU Training

If you have a GPU:

```bash
# Set device in environment
export DEVICE=cuda

# Or in Python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 📊 Dataset Requirements

### Minimum Dataset Sizes

- **Phishing Detection**: At least 1,000 samples per class (2,000 total minimum)
- **AI Detection**: At least 5,000 samples per class (10,000 total minimum)

### Recommended Dataset Sizes

- **Phishing Detection**: 10,000+ samples per class
- **AI Detection**: 50,000+ samples per class

### Data Quality Checklist

- [ ] Balanced classes (similar number of positive/negative samples)
- [ ] Clean text (remove excessive whitespace, special characters)
- [ ] Appropriate length (not too short, not too long)
- [ ] Diverse samples (different writing styles, topics)
- [ ] No data leakage (train/test split)

## 🐛 Troubleshooting

### Issue: "Dataset not found"
- Ensure CSV files are in `data/raw/` directory
- Check column names match: `text` and `label`

### Issue: "Out of memory"
- Reduce `per_device_train_batch_size`
- Use gradient accumulation
- Use a smaller model (e.g., `distilbert-base-uncased`)

### Issue: "Model not improving"
- Check data quality and balance
- Increase training epochs
- Adjust learning rate
- Try different model architectures

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

## 📄 License Notes

Always check dataset licenses before use:
- **CC0/Public Domain**: Free to use commercially
- **CC BY**: Attribution required
- **MIT**: Free to use with license notice
- **Academic**: May restrict commercial use
