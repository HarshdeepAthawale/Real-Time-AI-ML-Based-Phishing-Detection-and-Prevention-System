"""
Compare two model versions side-by-side
"""
import os
import sys
import json
import argparse
import logging
import boto3
from evaluate_model import evaluate_model, load_dataset_from_s3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_models(
    model_type: str,
    current_version: str,
    new_version: str,
    current_model_path: str,
    new_model_path: str,
    test_data: list,
    min_improvement: float = 0.01,
    max_fpr: float = 0.02
) -> dict:
    """Compare two model versions"""
    logger.info(f"Comparing {model_type} models: {current_version} vs {new_version}")
    
    # Evaluate both models
    current_metrics = evaluate_model(
        model_type,
        current_version,
        test_data,
        current_model_path
    )
    
    new_metrics = evaluate_model(
        model_type,
        new_version,
        test_data,
        new_model_path
    )
    
    # Calculate improvement
    f1_improvement = new_metrics['f1'] - current_metrics['f1']
    accuracy_improvement = new_metrics['accuracy'] - current_metrics['accuracy']
    
    # Determine if we should deploy
    should_deploy = (
        f1_improvement >= min_improvement and
        new_metrics['false_positive_rate'] <= max_fpr and
        new_metrics['f1'] >= current_metrics['f1']
    )
    
    reason = None
    if not should_deploy:
        if f1_improvement < min_improvement:
            reason = f"F1 improvement ({f1_improvement:.4f}) below threshold ({min_improvement})"
        elif new_metrics['false_positive_rate'] > max_fpr:
            reason = f"False positive rate ({new_metrics['false_positive_rate']:.4f}) exceeds maximum ({max_fpr})"
        elif new_metrics['f1'] < current_metrics['f1']:
            reason = f"New model F1 ({new_metrics['f1']:.4f}) is lower than current ({current_metrics['f1']:.4f})"
    
    comparison = {
        'model_type': model_type,
        'current_version': current_version,
        'new_version': new_version,
        'current_metrics': current_metrics,
        'new_metrics': new_metrics,
        'improvements': {
            'f1': float(f1_improvement),
            'accuracy': float(accuracy_improvement),
            'precision': float(new_metrics['precision'] - current_metrics['precision']),
            'recall': float(new_metrics['recall'] - current_metrics['recall']),
        },
        'should_deploy': should_deploy,
        'reason': reason,
    }
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Compare two model versions')
    parser.add_argument('--model-type', required=True, help='Model type (nlp/url/visual)')
    parser.add_argument('--current-version', required=True, help='Current model version')
    parser.add_argument('--new-version', required=True, help='New model version')
    parser.add_argument('--current-model-path', required=True, help='S3 path to current model')
    parser.add_argument('--new-model-path', required=True, help='S3 path to new model')
    parser.add_argument('--test-dataset', required=True, help='S3 path to test dataset')
    parser.add_argument('--output', required=True, help='Output file path for comparison JSON')
    parser.add_argument('--min-improvement', type=float, default=0.01, help='Minimum F1 improvement for deployment')
    parser.add_argument('--max-fpr', type=float, default=0.02, help='Maximum false positive rate')
    
    args = parser.parse_args()
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    # Load test dataset
    test_parts = args.test_dataset.replace('s3://', '').split('/', 1)
    test_bucket = test_parts[0]
    test_key = test_parts[1] if len(test_parts) > 1 else ''
    
    logger.info(f"Loading test dataset from s3://{test_bucket}/{test_key}")
    test_data = load_dataset_from_s3(s3_client, test_bucket, test_key)
    
    # Compare models
    comparison = compare_models(
        args.model_type,
        args.current_version,
        args.new_version,
        args.current_model_path,
        args.new_model_path,
        test_data,
        args.min_improvement,
        args.max_fpr
    )
    
    # Save comparison
    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison complete. Results saved to {args.output}")
    logger.info(f"F1 Improvement: {comparison['improvements']['f1']:.4f}")
    logger.info(f"Should Deploy: {comparison['should_deploy']}")
    if comparison['reason']:
        logger.info(f"Reason: {comparison['reason']}")


if __name__ == '__main__':
    main()
