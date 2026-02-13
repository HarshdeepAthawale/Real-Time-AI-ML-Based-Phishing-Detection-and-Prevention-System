"""
Training script for AI-generated content detector.
Downloads real datasets from Hugging Face for human vs AI text classification.

Usage:
    python scripts/train_ai_detector.py --epochs 5 --output-dir models/ai-detector
    python scripts/train_ai_detector.py --external-csv data/ai_text_dataset.csv
"""
import os
import json
import csv
import random
import argparse
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
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
    """Load AI-generated text detection dataset from Hugging Face."""
    try:
        from datasets import load_dataset

        datasets_to_try = [
            ("Hello-SimpleAI/HC3", "answer", None),
            ("aadityaubhat/GPT-wiki-intro", "generated_intro", None),
            ("artem9k/ai-text-detection-pile", "text", "label"),
        ]

        for ds_name, text_col, label_col in datasets_to_try:
            try:
                print(f"Attempting to load dataset: {ds_name}")
                ds = load_dataset(ds_name, split="train")
                samples = []

                for row in ds:
                    if label_col:
                        text = str(row.get(text_col, "")).strip()
                        label_raw = row.get(label_col, 0)
                        label = int(label_raw) if isinstance(label_raw, (int, float)) else (
                            1 if str(label_raw).lower() in ("ai", "generated", "1", "true") else 0
                        )
                    else:
                        # Dataset with separate human/AI columns
                        text = str(row.get(text_col, "")).strip()
                        label = 1  # AI-generated

                    if text and len(text) > 20:
                        samples.append((text[:1000], label))

                if len(samples) > 100:
                    print(f"Loaded {len(samples)} samples from {ds_name}")
                    return samples
            except Exception as e:
                print(f"  Could not load {ds_name}: {e}")
                continue

        return []
    except ImportError:
        print("Hugging Face datasets library not installed. Install: pip install datasets")
        return []


def load_external_dataset(dataset_path: str) -> List[Tuple[str, int]]:
    """Load labeled data from CSV. Format: text,label (0=human, 1=ai)"""
    samples = []
    if not os.path.exists(dataset_path):
        return samples

    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", row.get("content", ""))
            label_raw = row.get("label", row.get("is_ai", "0"))
            if isinstance(label_raw, str):
                label = 1 if label_raw.lower() in ("1", "ai", "generated", "true") else 0
            else:
                label = int(label_raw)
            if text and text.strip() and len(text.strip()) > 20:
                samples.append((text.strip()[:1000], label))

    print(f"Loaded {len(samples)} samples from {dataset_path}")
    return samples


def train(
    model_name="distilbert-base-uncased",
    output_dir="models/ai-detector",
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    external_csv=None,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load real datasets
    all_samples = []

    hf_samples = load_huggingface_dataset()
    all_samples.extend(hf_samples)

    if external_csv:
        ext = load_external_dataset(external_csv)
        all_samples.extend(ext)

    if len(all_samples) == 0:
        raise ValueError(
            "No training data available. Provide a dataset via:\n"
            "  --external-csv path/to/dataset.csv\n"
            "  Or install 'datasets' library: pip install datasets\n"
            "  CSV format: text,label (label: 0=human, 1=ai_generated)"
        )

    random.shuffle(all_samples)
    texts = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    print(f"\nTotal samples: {len(texts)}")
    print(f"  Human: {len(texts) - sum(labels)} ({(len(texts)-sum(labels))/len(texts)*100:.1f}%)")
    print(f"  AI:    {sum(labels)} ({sum(labels)/len(texts)*100:.1f}%)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "human_written", 1: "ai_generated"},
        label2id={"human_written": 0, "ai_generated": 1},
    )
    model.to(device)

    dataset = TextDataset(texts, labels, tokenizer)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

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

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | "
            f"Train Acc: {correct / total:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
            with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            with open(os.path.join(output_dir, "model_card.json"), "w") as f:
                json.dump({
                    "model_type": "ai-content-detector",
                    "base_model": model_name,
                    "num_labels": 2,
                    "labels": {"0": "human_written", "1": "ai_generated"},
                    "metrics": metrics,
                }, f, indent=2)
            print(f"  -> Saved best model (F1: {best_f1:.4f})")

    print(f"\nTraining complete! Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI content detector")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="models/ai-detector")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--external-csv", default=None, help="Path to labeled CSV (text,label)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        external_csv=args.external_csv,
        seed=args.seed,
    )
