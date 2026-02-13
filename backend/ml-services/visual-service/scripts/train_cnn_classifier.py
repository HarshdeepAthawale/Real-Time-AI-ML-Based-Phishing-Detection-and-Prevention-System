"""
Training script for CNN-based brand impersonation / phishing page classifier.
Downloads real phishing screenshot datasets from Hugging Face for visual phishing detection.

Usage:
    python scripts/train_cnn_classifier.py --epochs 10 --output-dir models/cnn-classifier
    python scripts/train_cnn_classifier.py --image-dir data/screenshots/
"""
import os
import json
import random
import argparse
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from PIL import Image


class PhishingCNN(nn.Module):
    """Lightweight CNN for visual phishing detection.

    Processes 3-channel (RGB) images resized to 224x224.
    Classifies as phishing (1) or legitimate (0).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Extract feature embeddings (pre-classifier)."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class ImageFolderDataset(Dataset):
    """Load images from a directory structure:
        image_dir/
            phishing/   (label=1)
            legitimate/  (label=0)
    """

    def __init__(self, image_dir: str, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        for label_name, label_id in [("legitimate", 0), ("phishing", 1)]:
            class_dir = os.path.join(image_dir, label_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    self.samples.append((os.path.join(class_dir, fname), label_id))

        if not self.samples:
            raise ValueError(
                f"No images found in {image_dir}. Expected subdirectories: phishing/ and legitimate/"
            )

        print(f"Loaded {len(self.samples)} images from {image_dir}")
        phishing_count = sum(1 for _, l in self.samples if l == 1)
        print(f"  Phishing: {phishing_count}, Legitimate: {len(self.samples) - phishing_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class HuggingFaceImageDataset(Dataset):
    """Load phishing screenshot dataset from Hugging Face."""

    def __init__(self, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self._load()

    def _load(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Hugging Face datasets library required. Install: pip install datasets"
            )

        datasets_to_try = [
            {
                "name": "phishintention/phishing-screenshot",
                "image_col": "image",
                "label_col": "label",
            },
            {
                "name": "biglab/webphish",
                "image_col": "image",
                "label_col": "label",
            },
            {
                "name": "Docta-ai/Phishing-Website-Screenshot",
                "image_col": "image",
                "label_col": "label",
            },
        ]

        for ds_info in datasets_to_try:
            try:
                print(f"Attempting to load dataset: {ds_info['name']}")
                ds = load_dataset(ds_info["name"], split="train")

                for row in ds:
                    img = row.get(ds_info["image_col"])
                    label_raw = row.get(ds_info["label_col"], 0)

                    if img is None:
                        continue

                    # Convert to PIL Image if not already
                    if not isinstance(img, Image.Image):
                        continue

                    img = img.convert("RGB")

                    if isinstance(label_raw, str):
                        label = 1 if label_raw.lower() in ("phishing", "1", "malicious", "bad") else 0
                    else:
                        label = int(label_raw)

                    self.images.append(img)
                    self.labels.append(label)

                if len(self.images) > 100:
                    print(f"Loaded {len(self.images)} images from {ds_info['name']}")
                    phishing_count = sum(self.labels)
                    print(f"  Phishing: {phishing_count}, Legitimate: {len(self.labels) - phishing_count}")
                    return
                else:
                    self.images.clear()
                    self.labels.clear()

            except Exception as e:
                print(f"  Could not load {ds_info['name']}: {e}")
                self.images.clear()
                self.labels.clear()
                continue

        if not self.images:
            raise ValueError(
                "No image datasets available. Provide screenshots via:\n"
                "  --image-dir path/to/images/ (with phishing/ and legitimate/ subdirs)\n"
                "  Or install 'datasets' library: pip install datasets"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size: int = 224):
    """Training and validation image transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def train(
    output_dir: str = "models/cnn-classifier",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_dir: Optional[str] = None,
    img_size: int = 224,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_transform, val_transform = get_transforms(img_size)

    # Load dataset from local directory or Hugging Face
    if image_dir and os.path.isdir(image_dir):
        print(f"Loading images from local directory: {image_dir}")
        dataset = ImageFolderDataset(image_dir, transform=train_transform)
    else:
        print("Loading images from Hugging Face datasets...")
        dataset = HuggingFaceImageDataset(transform=train_transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Model
    model = PhishingCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels if isinstance(labels, list) else labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | "
            f"Train Acc: {correct / total:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {"num_classes": 2, "input_size": img_size},
            }, os.path.join(output_dir, "model.pt"))

            metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
            with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            with open(os.path.join(output_dir, "model_card.json"), "w") as f:
                json.dump({
                    "model_type": "cnn-phishing-classifier",
                    "architecture": "PhishingCNN",
                    "input_shape": [3, img_size, img_size],
                    "num_classes": 2,
                    "labels": {"0": "legitimate", "1": "phishing"},
                    "metrics": metrics,
                }, f, indent=2)
            print(f"  -> Saved best model (F1: {best_f1:.4f})")

    print(f"\nTraining complete! Best Val F1: {best_f1:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train visual phishing CNN")
    parser.add_argument("--output-dir", default="models/cnn-classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-dir", default=None, help="Path to image dir with phishing/ and legitimate/ subdirs")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_dir=args.image_dir,
        img_size=args.img_size,
        seed=args.seed,
    )
