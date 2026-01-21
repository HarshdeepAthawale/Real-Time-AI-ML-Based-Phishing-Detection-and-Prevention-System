"""
Train CNN model for visual/brand impersonation detection
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrandImpersonationCNN(nn.Module):
    """CNN model for brand impersonation detection"""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ImageDataset(Dataset):
    """Dataset for image classification"""
    def __init__(self, data: list, transform=None):
        self.data = data
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image from S3 or use placeholder
        image = self.load_image(item)
        label = item.get('label', 0)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def load_image(self, item: dict):
        """Load image from item data"""
        # In production, download from S3
        # For now, create a placeholder image
        input_data = item.get('input_data', {})
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except:
                input_data = {}
        
        # Try to get image URL or path
        image_url = input_data.get('image_url') or input_data.get('screenshot_path')
        
        if image_url and image_url.startswith('s3://'):
            # Download from S3
            try:
                s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
                parts = image_url.replace('s3://', '').split('/', 1)
                bucket = parts[0]
                key = parts[1]
                
                response = s3_client.get_object(Bucket=bucket, Key=key)
                image_data = response['Body'].read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                return image
            except Exception as e:
                logger.warn(f"Failed to load image from S3: {e}")
        
        # Return placeholder image
        return Image.new('RGB', (224, 224), color=(128, 128, 128))


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


def train_model(
    train_data: list,
    val_data: list,
    output_path: str,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """Train the CNN model"""
    logger.info(f"Training CNN model")
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Create datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageDataset(train_data, transform=transform)
    val_dataset = ImageDataset(val_data, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrandImpersonationCNN(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
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
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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
    parser = argparse.ArgumentParser(description='Train CNN model for visual analysis')
    parser.add_argument('--dataset', required=True, help='S3 path to dataset (s3://bucket/key)')
    parser.add_argument('--output', required=True, help='Output S3 path (s3://bucket/prefix)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
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
            item['label'] = 1 if item.get('is_phishing', False) else 0
    
    # Create local output directory
    local_output = f"/tmp/cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        s3_model_prefix = f"{output_prefix}models/visual/{model_version}/"
        
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
