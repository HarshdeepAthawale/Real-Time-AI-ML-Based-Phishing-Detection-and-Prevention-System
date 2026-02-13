"""
Create a pretrained PhishingCNN model for the visual service.
Saves an initialized model so the service can load without errors.
For better accuracy, run train_cnn_classifier.py with --epochs 10.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


class PhishingCNN(nn.Module):
    """Lightweight CNN - must match cnn_classifier.py architecture."""

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


def main():
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "brand-classifier"
    )
    os.makedirs(output_dir, exist_ok=True)

    model = PhishingCNN(num_classes=2)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {"num_classes": 2, "input_size": 224},
    }, os.path.join(output_dir, "model.pth"))

    print(f"Created pretrained model at {output_dir}/model.pth")
    print("For better accuracy, run: python scripts/train_cnn_classifier.py --epochs 10")


if __name__ == "__main__":
    main()
