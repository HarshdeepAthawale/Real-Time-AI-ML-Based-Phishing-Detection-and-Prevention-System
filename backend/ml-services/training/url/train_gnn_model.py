"""
Train GNN model for URL/domain analysis
"""
import os
import sys
import json
import argparse
import logging
import boto3
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleGNNClassifier(nn.Module):
    """Simple GNN classifier for domain graphs"""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return torch.log_softmax(x, dim=1)


def load_dataset_from_s3(s3_client, bucket: str, key: str) -> list:
    """Load dataset from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def create_graph_from_data(item: dict) -> Data:
    """Create PyTorch Geometric graph from data item"""
    # Extract features
    input_data = item.get('input_data', {})
    if isinstance(input_data, str):
        input_data = json.loads(input_data) if input_data.startswith('{') else {}
    
    # Create node features (simplified - in production, use actual graph builder)
    num_nodes = 1
    node_features = np.array([
        [
            len(input_data.get('domain', '')),
            input_data.get('reputation_score', 0) / 100.0,
            input_data.get('age_days', 0) / 3650.0,
            1.0 if input_data.get('is_suspicious', False) else 0.0,
            1.0 if input_data.get('has_ssl', False) else 0.0,
            input_data.get('redirect_count', 0) / 10.0,
            input_data.get('suspicious_keywords', 0) / 10.0,
            input_data.get('homoglyph_score', 0.0),
            input_data.get('typosquatting_score', 0.0),
            input_data.get('domain_age_score', 0.0),
        ]
    ])
    
    # Create edge index (self-loop for single node)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    
    # Create graph
    graph = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([item.get('label', 0)], dtype=torch.long)
    )
    
    return graph


def train_model(
    train_data: list,
    val_data: list,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """Train the GNN model"""
    logger.info(f"Training GNN model")
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Create graphs
    train_graphs = [create_graph_from_data(item) for item in train_data]
    val_graphs = [create_graph_from_data(item) for item in val_data]
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGNNClassifier(input_dim=10, hidden_dim=64, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
            if batch.x.size(0) == 0:
                continue
                
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if batch.x.size(0) == 0:
                    continue
                    
                output = model(batch)
                loss = criterion(output, batch.y.squeeze())
                val_loss += loss.item()
                
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.squeeze().cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
        )
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_path, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_path, 'best_model.pt')))
    torch.save(model.state_dict(), os.path.join(output_path, 'model.pt'))
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            if batch.x.size(0) == 0:
                continue
            output = model(batch)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
    }
    
    return metrics


def upload_to_s3(s3_client, bucket: str, local_path: str, s3_prefix: str):
    """Upload model files to S3"""
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            s3_key = f"{s3_prefix}{relative_path}"
            
            s3_client.upload_file(local_file, bucket, s3_key)
            logger.info(f"Uploaded {local_file} to s3://{bucket}/{s3_key}")


def main():
    parser = argparse.ArgumentParser(description='Train GNN model for URL analysis')
    parser.add_argument('--dataset', required=True, help='S3 path to dataset (s3://bucket/key)')
    parser.add_argument('--output', required=True, help='Output S3 path (s3://bucket/prefix)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Parse S3 paths
    dataset_parts = args.dataset.replace('s3://', '').split('/', 1)
    dataset_bucket = dataset_parts[0]
    dataset_prefix = dataset_parts[1] if len(dataset_parts) > 1 else ''
    
    output_parts = args.output.replace('s3://', '').split('/', 1)
    output_bucket = output_parts[0]
    output_prefix = output_parts[1] if len(output_parts) > 1 else ''
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    # Load datasets
    logger.info(f"Loading datasets from s3://{dataset_bucket}/{dataset_prefix}")
    train_data = load_dataset_from_s3(s3_client, dataset_bucket, f"{dataset_prefix}train.json")
    val_data = load_dataset_from_s3(s3_client, dataset_bucket, f"{dataset_prefix}val.json")
    
    # Ensure labels exist
    for item in train_data + val_data:
        if 'label' not in item:
            item['label'] = 1 if item.get('threat_type') else 0
    
    # Create local output directory
    local_output = f"/tmp/gnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(local_output, exist_ok=True)
    
    try:
        # Train model
        metrics = train_model(
            train_data,
            val_data,
            local_output,
            args.epochs,
            args.batch_size,
            args.learning_rate
        )
        
        # Save metrics
        metrics_file = os.path.join(local_output, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Upload to S3
        model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_model_prefix = f"{output_prefix}models/url/{model_version}/"
        
        logger.info(f"Uploading model to s3://{output_bucket}/{s3_model_prefix}")
        upload_to_s3(s3_client, output_bucket, local_output, s3_model_prefix)
        
        logger.info("Training completed successfully")
        
    finally:
        # Cleanup
        import shutil
        if os.path.exists(local_output):
            shutil.rmtree(local_output)


if __name__ == '__main__':
    main()
