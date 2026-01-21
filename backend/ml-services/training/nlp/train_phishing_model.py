"""
Train phishing detection model using BERT/RoBERTa
"""
import os
import sys
import json
import argparse
import logging
import boto3
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def prepare_datasets(train_data: list, val_data: list, tokenizer, max_length: int = 512):
    """Prepare datasets for training"""
    def tokenize_function(examples):
        texts = examples['text'] if isinstance(examples['text'], list) else [examples['text']]
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, val_dataset


def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
    }


def train_model(
    train_data: list,
    val_data: list,
    output_path: str,
    model_name: str = 'roberta-base',
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """Train the model"""
    logger.info(f"Training model: {model_name}")
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(train_data, val_data, tokenizer)
    
    # Set labels
    train_dataset = train_dataset.map(lambda x: {'labels': x['label']})
    val_dataset = val_dataset.map(lambda x: {'labels': x['label']})
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_path}/logs',
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        learning_rate=learning_rate,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Model saved to {output_path}")
    logger.info(f"Evaluation metrics: {eval_results}")
    
    return eval_results


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
    parser = argparse.ArgumentParser(description='Train phishing detection model')
    parser.add_argument('--dataset', required=True, help='S3 path to dataset (s3://bucket/key)')
    parser.add_argument('--output', required=True, help='Output S3 path (s3://bucket/prefix)')
    parser.add_argument('--model-name', default='roberta-base', help='Base model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    
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
    
    # Ensure data has required fields
    for item in train_data + val_data:
        if 'text' not in item:
            item['text'] = item.get('input_data', {}).get('text', '') if isinstance(item.get('input_data'), dict) else str(item.get('input_data', ''))
        if 'label' not in item:
            item['label'] = 1 if item.get('feedback_type') == 'true_positive' else 0
    
    # Create local output directory
    local_output = f"/tmp/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(local_output, exist_ok=True)
    
    try:
        # Train model
        metrics = train_model(
            train_data,
            val_data,
            local_output,
            args.model_name,
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
        s3_model_prefix = f"{output_prefix}models/nlp/{model_version}/"
        
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
