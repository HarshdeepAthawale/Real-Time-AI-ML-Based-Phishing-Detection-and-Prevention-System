"""
Prepare and split datasets for training
"""
import os
import sys
import json
import boto3
import pandas as pd
import argparse
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset_from_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    """Load dataset JSON from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame([data])
    except Exception as e:
        logger.error(f"Failed to load dataset from s3://{bucket}/{key}: {e}")
        raise


def prepare_splits(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation, and test sets"""
    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'] if 'label' in df.columns else None,
        random_state=42
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        stratify=train_val['label'] if 'label' in train_val.columns else None,
        random_state=42
    )
    
    logger.info(f"Dataset splits - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return train, val, test


def save_to_s3(s3_client, bucket: str, key: str, df: pd.DataFrame):
    """Save DataFrame to S3 as JSON"""
    data = df.to_dict('records')
    json_str = json.dumps(data, indent=2, default=str)
    
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json_str.encode('utf-8'),
        ContentType='application/json'
    )
    
    logger.info(f"Saved to s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--dataset', required=True, help='S3 path to dataset (s3://bucket/key)')
    parser.add_argument('--output-prefix', required=True, help='S3 prefix for output files')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size (default: 0.1)')
    
    args = parser.parse_args()
    
    # Parse S3 path
    if args.dataset.startswith('s3://'):
        parts = args.dataset.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
    else:
        raise ValueError("Dataset path must be an S3 path (s3://bucket/key)")
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    # Load dataset
    logger.info(f"Loading dataset from s3://{bucket}/{key}")
    df = load_dataset_from_s3(s3_client, bucket, key)
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Ensure required columns
    if 'text' not in df.columns and 'label' not in df.columns:
        # Try to extract from input_data
        if 'input_data' in df.columns:
            df['text'] = df['input_data'].apply(
                lambda x: x.get('text') or x.get('body') or x.get('content') if isinstance(x, dict) else str(x)
            )
        if 'feedback_type' in df.columns:
            df['label'] = df['feedback_type'].apply(
                lambda x: 1 if x == 'true_positive' else 0
            )
    
    # Remove invalid rows
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.len() > 10]
    
    # Prepare splits
    train, val, test = prepare_splits(df, args.test_size, args.val_size)
    
    # Save splits to S3
    save_to_s3(s3_client, bucket, f"{args.output_prefix}train.json", train)
    save_to_s3(s3_client, bucket, f"{args.output_prefix}val.json", val)
    save_to_s3(s3_client, bucket, f"{args.output_prefix}test.json", test)
    
    # Save metadata
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train),
        'val_samples': len(val),
        'test_samples': len(test),
        'label_distribution': {
            'train': train['label'].value_counts().to_dict(),
            'val': val['label'].value_counts().to_dict(),
            'test': test['label'].value_counts().to_dict(),
        }
    }
    
    metadata_key = f"{args.output_prefix}metadata.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode('utf-8'),
        ContentType='application/json'
    )
    
    logger.info(f"Dataset preparation complete. Metadata saved to s3://{bucket}/{metadata_key}")


if __name__ == '__main__':
    main()
