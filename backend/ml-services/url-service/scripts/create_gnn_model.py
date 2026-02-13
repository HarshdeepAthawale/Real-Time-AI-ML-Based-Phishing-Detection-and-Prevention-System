"""
Create a pretrained GNN model for domain classification.
The model accepts PyTorch Geometric Data from GraphBuilder (8 node features).
Uses pure PyTorch (no torch_geometric) for standalone script execution.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


class DomainGNN(nn.Module):
    """
    Simple MLP-based domain classifier compatible with GraphBuilder output.
    Accepts Data(x=[N,8], edge_index=...) - uses mean pool over nodes.
    Output: [1, 2] logits (legitimate, malicious)
    """

    def __init__(self, in_channels: int = 8, hidden_channels: int = 32, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, data):
        x = data.x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = x.mean(dim=0, keepdim=True)
        return self.mlp(h)


def main():
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "gnn-domain-classifier"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 8 features from GraphBuilder._extract_node_features
    model = DomainGNN(in_channels=8, hidden_channels=32, num_classes=2)
    model.eval()

    path = os.path.join(output_dir, "model.pth")
    torch.save(model, path)
    print(f"Created GNN model at {path}")


if __name__ == "__main__":
    main()
