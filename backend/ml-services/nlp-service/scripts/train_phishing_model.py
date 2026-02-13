"""
Training script for BERT-based phishing classifier.
Downloads real phishing email datasets from Hugging Face,
fine-tunes a DistilBERT model, and saves for inference.

Usage:
    python scripts/train_phishing_model.py --epochs 5 --batch-size 16 --output-dir models/phishing-detector
    python scripts/train_phishing_model.py --external-csv data/custom_dataset.csv
"""
import os
import sys
import json
import csv
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import numpy as np


class PhishingDataset(Dataset):
    """PyTorch dataset for phishing text classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_huggingface_dataset() -> List[Tuple[str, int]]:
    """Load phishing email dataset from Hugging Face datasets hub."""
    try:
        from datasets import load_dataset

        # Try multiple real phishing datasets from Hugging Face
        datasets_to_try = [
            ("ealvaradob/phishing-dataset", "text", "label"),
            ("ColumbiaNYU/phishing-email", "body", "label"),
            ("pirocheto/phishing-url", "url", "status"),
        ]

        for ds_name, text_col, label_col in datasets_to_try:
            try:
                print(f"Attempting to load dataset: {ds_name}")
                ds = load_dataset(ds_name, split="train")
                samples = []
                for row in ds:
                    text = str(row.get(text_col, "")).strip()
                    label_raw = row.get(label_col, 0)
                    if isinstance(label_raw, str):
                        label = 1 if label_raw.lower() in ("phishing", "1", "spam", "malicious", "bad") else 0
                    else:
                        label = int(label_raw)
                    if text and len(text) > 10:
                        samples.append((text, label))
                if len(samples) > 100:
                    print(f"Loaded {len(samples)} samples from {ds_name}")
                    return samples
            except Exception as e:
                print(f"  Could not load {ds_name}: {e}")
                continue

        print("No Hugging Face datasets available. Using external CSV or raising error.")
        return []

    except ImportError:
        print("Hugging Face datasets library not installed. Install with: pip install datasets")
        return []


def load_external_dataset(dataset_path: str) -> List[Tuple[str, int]]:
    """Load labeled data from CSV file.

    Expected CSV format: text,label (where label is 0 or 1)
    Also supports: content,is_phishing or email,label
    """
    samples = []
    if not os.path.exists(dataset_path):
        print(f"External CSV not found: {dataset_path}")
        return samples

    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", row.get("content", row.get("email", row.get("body", ""))))
            label_raw = row.get("label", row.get("is_phishing", row.get("status", "0")))
            if isinstance(label_raw, str):
                label = 1 if label_raw.lower() in ("1", "phishing", "spam", "malicious", "true") else 0
            else:
                label = int(label_raw)
            if text and text.strip() and len(text.strip()) > 10:
                samples.append((text.strip(), label))

    print(f"Loaded {len(samples)} samples from {dataset_path}")
    return samples


def prepare_data(
    external_csv: str = None,
) -> Tuple[List[str], List[int]]:
    """Prepare training data from real datasets."""
    all_samples = []

    # 1. Try loading from Hugging Face
    hf_samples = load_huggingface_dataset()
    all_samples.extend(hf_samples)

    # 2. Load external CSV if provided
    if external_csv:
        ext = load_external_dataset(external_csv)
        all_samples.extend(ext)

    if len(all_samples) == 0:
        raise ValueError(
            "No training data available. Provide a dataset via:\n"
            "  --external-csv path/to/dataset.csv\n"
            "  Or install 'datasets' library: pip install datasets\n"
            "  CSV format: text,label (label: 0=legitimate, 1=phishing)"
        )

    random.shuffle(all_samples)
    texts = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    print(f"\nTotal samples: {len(texts)}")
    print(f"  Phishing:   {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Legitimate: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

    return texts, labels


def train(
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/phishing-detector",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    external_csv: str = None,
    seed: int = 42,
):
    """Fine-tune a transformer model for phishing detection."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Prepare data
    texts, labels = prepare_data(external_csv)

    # Load tokenizer and model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "legitimate", 1: "phishing"},
        label2id={"legitimate": 0, "phishing": 1},
    )
    model.to(device)

    # Create dataset and split
    dataset = PhishingDataset(texts, labels, tokenizer, max_length)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    # Training loop
    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)

        # Validation
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_model(model, tokenizer, output_dir, val_metrics, model_name)
            print(f"  -> Saved best model (F1: {best_f1:.4f})")

    print(f"\nTraining complete! Best Val F1: {best_f1:.4f}")
    print(f"Model saved to: {output_dir}")

    return best_f1


def evaluate(model, data_loader, device) -> Dict:
    """Evaluate model on validation/test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1
    )

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        "confusion_matrix": cm.tolist(),
    }


def save_model(model, tokenizer, output_dir: str, metrics: Dict, base_model: str):
    """Save model, tokenizer, and metrics to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metrics
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save model card
    model_card = {
        "model_type": "phishing-classifier",
        "base_model": base_model,
        "num_labels": 2,
        "labels": {"0": "legitimate", "1": "phishing"},
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()},
    }
    with open(os.path.join(output_dir, "model_card.json"), "w") as f:
        json.dump(model_card, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phishing classifier")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="models/phishing-detector")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--external-csv", default=None, help="Path to labeled CSV dataset")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        external_csv=args.external_csv,
        seed=args.seed,
    )
