"""
Collect and format feedback data from S3 for training
"""
import os
import sys
import json
import boto3
import pandas as pd
from typing import List, Dict, Any
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_from_s3(s3_client, bucket: str, prefix: str) -> List[Dict[str, Any]]:
    """Download all JSON files from S3 prefix"""
    all_data = []
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            if obj['Key'].endswith('.json'):
                try:
                    response = s3_client.get_object(Bucket=bucket, Key=obj['Key'])
                    content = response['Body'].read().decode('utf-8')
                    data = json.loads(content)
                    
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                        
                    logger.info(f"Downloaded {obj['Key']}: {len(data) if isinstance(data, list) else 1} records")
                except Exception as e:
                    logger.error(f"Failed to download {obj['Key']}: {e}")
    
    return all_data


def format_feedback_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format feedback data for training"""
    formatted = []
    
    for record in data:
        # Extract text from input_data
        text = ''
        if 'input_data' in record:
            if isinstance(record['input_data'], dict):
                text = record['input_data'].get('text') or record['input_data'].get('body') or record['input_data'].get('content') or ''
            elif isinstance(record['input_data'], str):
                text = record['input_data']
        
        # Determine label from feedback_type
        label = 0  # Default to legitimate
        if record.get('feedback_type') == 'true_positive':
            label = 1  # Phishing
        elif record.get('feedback_type') == 'false_positive':
            label = 0  # Legitimate (false positive means it was incorrectly flagged)
        elif record.get('feedback_type') == 'false_negative':
            label = 1  # Phishing (false negative means it was missed)
        
        formatted.append({
            'text': text,
            'label': label,
            'feedback_type': record.get('feedback_type'),
            'detection_id': record.get('detection_id'),
            'confidence_score': record.get('confidence_score', 0),
        })
    
    df = pd.DataFrame(formatted)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 10]
    
    logger.info(f"Formatted {len(df)} feedback records")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Collect feedback data from S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--prefix', default='training-data/feedback/', help='S3 prefix')
    parser.add_argument('--output', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    # Download data
    logger.info(f"Downloading feedback data from s3://{args.bucket}/{args.prefix}")
    data = download_from_s3(s3_client, args.bucket, args.prefix)
    
    if not data:
        logger.warning("No data found")
        return
    
    # Format data
    df = format_feedback_data(data)
    
    # Save to output
    df.to_csv(args.output, index=False)
    logger.info(f"Saved formatted data to {args.output}")


if __name__ == '__main__':
    main()
