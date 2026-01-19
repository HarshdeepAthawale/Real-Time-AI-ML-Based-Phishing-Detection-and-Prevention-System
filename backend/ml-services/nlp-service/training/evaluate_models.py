"""
Comprehensive model evaluation module
Calculate accuracy, precision, recall, F1-score, confusion matrix, ROC curve, etc.
"""
import os
import sys
import json
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
from training.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model_comprehensive(
    model_path: str,
    test_dataset: Dataset,
    model_name: str = "distilbert-base-uncased",
    device: str = "cpu"
) -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        model_path: Path to trained model
        test_dataset: Test dataset
        model_name: Base model name
        device: Device to run evaluation on
        
    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info(f"Evaluating model from {model_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Create trainer for evaluation
    trainer = Trainer(model=model, tokenizer=tokenizer)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    pred_probs = np.softmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Classification report
    class_report = classification_report(
        true_labels, pred_labels,
        target_names=['Class 0', 'Class 1'],
        output_dict=True
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_metrics': {
            'class_0': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1': float(f1[0]),
                'support': int(support[0])
            },
            'class_1': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1': float(f1[1]),
                'support': int(support[1])
            }
        },
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
        'classification_report': class_report
    }
    
    return metrics

def plot_confusion_matrix(cm: np.ndarray, output_path: str, class_names: list = None):
    """Plot and save confusion matrix"""
    if class_names is None:
        class_names = ['Legitimate', 'Phishing']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_path: str):
    """Plot and save ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved ROC curve to {output_path}")

def generate_evaluation_report(
    metrics: Dict,
    model_name: str,
    output_dir: str,
    class_names: list = None
) -> str:
    """
    Generate comprehensive evaluation report (JSON + Markdown)
    
    Args:
        metrics: Evaluation metrics dictionary
        model_name: Name of the model
        output_dir: Directory to save reports
        class_names: Names of classes
        
    Returns:
        Path to markdown report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if class_names is None:
        class_names = ['Legitimate', 'Phishing']
    
    # Save JSON report
    json_path = os.path.join(output_dir, f"{model_name}_evaluation.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved JSON report to {json_path}")
    
    # Generate Markdown report
    md_path = os.path.join(output_dir, f"{model_name}_evaluation.md")
    
    with open(md_path, 'w') as f:
        f.write(f"# Model Evaluation Report: {model_name}\n\n")
        f.write("## Summary Metrics\n\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
        f.write(f"- **Precision (Macro)**: {metrics['precision_macro']:.4f}\n")
        f.write(f"- **Recall (Macro)**: {metrics['recall_macro']:.4f}\n")
        f.write(f"- **F1-Score (Macro)**: {metrics['f1_macro']:.4f}\n")
        f.write(f"- **ROC AUC**: {metrics['roc_auc']:.4f}\n\n")
        
        f.write("## Per-Class Metrics\n\n")
        for i, class_name in enumerate(class_names):
            class_key = f'class_{i}'
            if class_key in metrics['per_class_metrics']:
                class_metrics = metrics['per_class_metrics'][class_key]
                f.write(f"### {class_name}\n\n")
                f.write(f"- Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"- Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"- F1-Score: {class_metrics['f1']:.4f}\n")
                f.write(f"- Support: {class_metrics['support']}\n\n")
        
        f.write("## Confusion Matrix\n\n")
        cm = np.array(metrics['confusion_matrix'])
        f.write(f"| | {class_names[0]} | {class_names[1]} |\n")
        f.write("| --- | --- | --- |\n")
        for i, class_name in enumerate(class_names):
            f.write(f"| **{class_name}** | {cm[i][0]} | {cm[i][1]} |\n")
        f.write("\n")
    
    logger.info(f"Saved Markdown report to {md_path}")
    
    return md_path

def compare_models(model_metrics: Dict[str, Dict]) -> str:
    """
    Compare multiple models and generate comparison report
    
    Args:
        model_metrics: Dictionary mapping model names to their metrics
        
    Returns:
        Path to comparison report
    """
    output_dir = config.METRICS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_path = os.path.join(output_dir, "model_comparison.md")
    
    with open(comparison_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        for model_name, metrics in model_metrics.items():
            f.write(f"| {model_name} | "
                   f"{metrics['accuracy']:.4f} | "
                   f"{metrics['precision_macro']:.4f} | "
                   f"{metrics['recall_macro']:.4f} | "
                   f"{metrics['f1_macro']:.4f} | "
                   f"{metrics['roc_auc']:.4f} |\n")
    
    logger.info(f"Saved comparison report to {comparison_path}")
    return comparison_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to test dataset CSV")
    parser.add_argument("--output", type=str, help="Output directory for reports")
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased", help="Base model name")
    
    args = parser.parse_args()
    
    output_dir = args.output or config.METRICS_DIR
    
    # Load test dataset
    df = pd.read_csv(args.dataset)
    test_dataset = Dataset.from_pandas(df)
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512
        )
    
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Evaluate
    metrics = evaluate_model_comprehensive(
        model_path=args.model,
        test_dataset=test_dataset,
        model_name=args.base_model
    )
    
    # Generate reports
    model_name = os.path.basename(args.model)
    generate_evaluation_report(metrics, model_name, output_dir)
    
    logger.info("Evaluation completed!")
