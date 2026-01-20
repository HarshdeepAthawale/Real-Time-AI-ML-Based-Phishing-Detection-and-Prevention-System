"""
Training script for GNN Domain Classifier
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import json
import os
from typing import List, Dict
import argparse
from datetime import datetime

# Import model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.gnn_classifier import DomainGNNClassifier
from src.graph.graph_builder import GraphBuilder

def generate_synthetic_data(num_samples: int = 1000) -> List[Data]:
    """Generate synthetic training data"""
    graph_builder = GraphBuilder()
    data_list = []
    
    for i in range(num_samples):
        # Create synthetic domain data
        domains = [{
            'id': f'domain_{i}',
            'domain': f'example{i}.com',
            'reputation_score': np.random.uniform(0, 100),
            'age_days': np.random.randint(0, 3650),
            'is_malicious': np.random.random() > 0.7,
            'is_suspicious': np.random.random() > 0.5
        }]
        
        # Create synthetic relationships
        relationships = []
        if i > 0:
            relationships.append({
                'source_domain_id': f'domain_{i-1}',
                'target_domain_id': f'domain_{i}',
                'relationship_type': 'redirects_to',
                'strength': np.random.uniform(0.5, 1.0)
            })
        
        # Build graph
        graph_data = graph_builder.build_domain_graph(domains, relationships)
        
        # Add label (0 = legitimate, 1 = malicious)
        label = 1 if domains[0]['is_malicious'] else 0
        graph_data.y = torch.tensor([label], dtype=torch.long)
        
        data_list.append(graph_data)
    
    return data_list

def train_model(
    model: DomainGNNClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001
):
    """Train the GNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Handle empty graphs
            if batch.x.size(0) == 0:
                continue
            
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                if batch.x.size(0) == 0:
                    continue
                
                output = model(batch)
                loss = criterion(output, batch.y)
                val_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(batch.y).sum().item()
                total += batch.y.size(0)
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = correct / total if total > 0 else 0
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {accuracy:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train GNN Domain Classifier')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--train-samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=200, help='Number of validation samples')
    parser.add_argument('--output-dir', type=str, default='./models/gnn-domain-classifier-v1', help='Output directory')
    
    args = parser.parse_args()
    
    print("Generating synthetic training data...")
    train_data = generate_synthetic_data(args.train_samples)
    val_data = generate_synthetic_data(args.val_samples)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    print("Initializing model...")
    model = DomainGNNClassifier(input_dim=5, hidden_dim=64, num_classes=2)
    
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'model.pt')
    trained_model.save(model_path)
    
    # Save training metadata
    metadata = {
        'model_type': 'DomainGNNClassifier',
        'input_dim': 5,
        'hidden_dim': 64,
        'num_classes': 2,
        'training_date': datetime.now().isoformat(),
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'train_samples': args.train_samples,
        'val_samples': args.val_samples
    }
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print("Training completed!")

if __name__ == "__main__":
    main()
