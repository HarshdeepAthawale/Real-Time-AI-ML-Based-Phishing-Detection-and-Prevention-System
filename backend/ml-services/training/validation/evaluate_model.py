"""
Evaluate a model on a test dataset
"""
import os
import sys
import json
import argparse
import logging
import boto3
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

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


def load_model_metrics_from_s3(s3_client, bucket: str, key: str) -> dict:
    """Load model metrics from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        metrics = json.loads(content)
        return metrics
    except Exception as e:
        logger.warn(f"Failed to load metrics from S3: {e}")
        return {}


def evaluate_model(
    model_type: str,
    model_version: str,
    test_data: list,
    model_path_s3: str
) -> dict:
    """
    Evaluate a model on test data
    
    In production, this would:
    1. Download model from S3
    2. Load model weights
    3. Run inference on test data
    4. Calculate metrics
    
    For now, we'll use stored metrics if available, or calculate from predictions
    """
    logger.info(f"Evaluating {model_type} model v{model_version}")
    
    # Try to load metrics from model directory
    s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    # Parse S3 path
    if model_path_s3.startswith('s3://'):
        parts = model_path_s3.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
    else:
        bucket = os.getenv('S3_BUCKET_MODELS', '')
        prefix = model_path_s3
    
    metrics_key = f"{prefix}metrics.json"
    metrics = load_model_metrics_from_s3(s3_client, bucket, metrics_key)
    
    if metrics:
        logger.info("Using stored metrics from model")
        return metrics
    
    # If no stored metrics, calculate from test data predictions
    # In production, this would run actual model inference
    logger.warn("No stored metrics found. Using placeholder evaluation.")
    
    # Extract labels from test data
    labels = [item.get('label', 0) for item in test_data]
    
    # Generate placeholder predictions (in production, use actual model)
    # For now, use a simple heuristic based on confidence scores
    predictions = []
    for item in test_data:
        confidence = item.get('confidence_score', 0.5)
        if confidence > 0.7:
            predictions.append(1)  # Phishing
        else:
            predictions.append(0)  # Legitimate
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'total_samples': len(test_data),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    parser.add_argument('--model-type', required=True, help='Model type (nlp/url/visual)')
    parser.add_argument('--model-version', required=True, help='Model version')
    parser.add_argument('--model-path', required=True, help='S3 path to model (s3://bucket/prefix)')
    parser.add_argument('--test-dataset', required=True, help='S3 path to test dataset (s3://bucket/key)')
    parser.add_argument('--output', required=True, help='Output file path for metrics JSON')
    
    args = parser.parse_args()
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    # Load test dataset
    test_parts = args.test_dataset.replace('s3://', '').split('/', 1)
    test_bucket = test_parts[0]
    test_key = test_parts[1] if len(test_parts) > 1 else ''
    
    logger.info(f"Loading test dataset from s3://{test_bucket}/{test_key}")
    test_data = load_dataset_from_s3(s3_client, test_bucket, test_key)
    
    # Evaluate model
    metrics = evaluate_model(
        args.model_type,
        args.model_version,
        test_data,
        args.model_path
    )
    
    # Save metrics
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation complete. Metrics saved to {args.output}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
