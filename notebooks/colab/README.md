# Google Colab Training Notebooks

These notebooks train the ML models used by the Phishing Detection system.
Run them on Google Colab with a **T4 GPU** for best performance.

## Notebooks

| # | Notebook | Model | Target | Output Directory |
|---|----------|-------|--------|------------------|
| 01 | `01_train_phishing_classifier.ipynb` | DistilBERT phishing classifier | F1 >= 0.95 | `nlp-service/models/phishing-detector/` |
| 02 | `02_train_ai_detector.ipynb` | DistilBERT AI-text detector | F1 >= 0.90 | `nlp-service/models/ai-detector/` |
| 03 | `03_train_gnn_url_classifier.ipynb` | DomainGNN URL classifier | AUC >= 0.95 | `url-service/models/gnn-domain-classifier/` |
| 04 | `04_train_cnn_visual_classifier.ipynb` | PhishingCNN screenshot classifier | F1 >= 0.93 | `visual-service/models/brand-classifier/` |

## Workflow

1. Open notebook in Google Colab
2. Set runtime to **GPU** (Runtime -> Change runtime type -> T4 GPU)
3. **(Optional)** Add your Hugging Face token for gated datasets and higher rate limits: click the **key icon** in the left sidebar → **Secrets** → **Add new secret** → Name: `HF_TOKEN`, Value: your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Run all cells
5. Download the output zip file
6. Run the installer script:
   ```bash
   ./scripts/download-trained-models.sh --from-dir /path/to/downloaded/models
   ```

## Output Formats

- **NLP models**: HuggingFace format (config.json, model.safetensors, tokenizer.json, etc.)
- **GNN model**: Full pickled PyTorch model (model.pth)
- **CNN model**: State dict checkpoint (model.pth with model_state_dict key)

Each model also outputs `training_metrics.json` and `model_card.json`.
