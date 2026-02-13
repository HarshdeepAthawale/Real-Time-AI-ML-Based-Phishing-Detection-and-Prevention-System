# NLP Service Training Scripts

## Phishing Classifier Training

### Prerequisites

```bash
pip install -r requirements.txt
# datasets library is required for Hugging Face datasets
```

### Training Commands

**Using Hugging Face datasets (recommended):**
```bash
cd backend/ml-services/nlp-service
python scripts/train_phishing_model.py --epochs 5 --batch-size 16
```

**Using external CSV:**
```bash
python scripts/train_phishing_model.py \
  --external-csv path/to/dataset.csv \
  --epochs 5 \
  --output-dir models/phishing-detector
```

**Minimal run (no dataset - uses built-in fallback):**
```bash
python scripts/train_phishing_model.py --epochs 3
```

### Dataset Formats

**Hugging Face:** The script tries these datasets automatically:
- `ealvaradob/phishing-dataset`
- `ColumbiaNYU/phishing-email`
- `pirocheto/phishing-url`

**CSV format:** `text,label` or `content,is_phishing` where label is 0 (legitimate) or 1 (phishing).

**Public phishing datasets:**
- [PhishIntention](https://github.com/APWG/phishintention)
- [SOREB-Phish](https://github.com/soreb-phish)
- [APWG eCrime dataset](https://apwg.org/ecrime-research/)

### Output

Model is saved to `models/phishing-detector` (or `$MODEL_DIR/phishing-detector`):
- `config.json`, `pytorch_model.bin` - model weights
- `tokenizer.json`, `vocab.txt` - tokenizer
- `training_metrics.json` - validation metrics
- `model_card.json` - model metadata

### Validation

After training, validate the model:
```bash
python scripts/validate_models.py
```
