#!/bin/bash
# Setup all ML models for the phishing detection system.
# Run from project root: ./scripts/setup-ml-models.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Setting up ML models ==="

# 1. NLP - Phishing detector (already copied from external BERT-YAKE)
echo "[1/4] NLP phishing-detector: using BERT model from external/Phishing-Detection-BERT-YAKE"
if [ -d "external/Phishing-Detection-BERT-YAKE/models/bert_model" ]; then
  mkdir -p backend/ml-services/nlp-service/models/phishing-detector
  cp -r external/Phishing-Detection-BERT-YAKE/models/bert_model/* backend/ml-services/nlp-service/models/phishing-detector/
  # Ensure config has num_labels for BertForSequenceClassification
  python3 -c "
import json, os
p = 'backend/ml-services/nlp-service/models/phishing-detector/config.json'
if os.path.exists(p):
  with open(p) as f: c = json.load(f)
  if 'num_labels' not in c:
    c['num_labels'] = 2
    c['id2label'] = {'0': 'legitimate', '1': 'phishing'}
    c['label2id'] = {'legitimate': 0, 'phishing': 1}
    with open(p, 'w') as f: json.dump(c, f, indent=2)
    print('  Added num_labels to config')
" 2>/dev/null || true
  echo "  Done"
else
  echo "  Skip (external repo not cloned). Run: git clone ... external/Phishing-Detection-BERT-YAKE"
fi

# 2. NLP - AI detector
echo "[2/4] NLP ai-detector: downloading from HuggingFace..."
(cd backend/ml-services/nlp-service && python scripts/download_ai_detector.py) || echo "  Skip (requires: pip install transformers torch)"

# 3. Visual - Brand classifier
echo "[3/4] Visual brand-classifier: creating pretrained model..."
python backend/ml-services/visual-service/scripts/create_pretrained_model.py || \
  (cd backend/ml-services/visual-service && python scripts/create_pretrained_model.py)

# 4. URL - GNN
echo "[4/4] URL gnn-domain-classifier: creating GNN model..."
python backend/ml-services/url-service/scripts/create_gnn_model.py || \
  (cd backend/ml-services/url-service && python scripts/create_gnn_model.py)

echo ""
echo "=== ML models setup complete ==="
echo "Models are in backend/ml-services/*/models/"
echo "Run with Docker: cd backend && docker-compose up"
