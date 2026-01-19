"""
Training script for phishing detection model
Fine-tune BERT/RoBERTa on phishing email dataset
"""
import os
import sys
import json
import logging
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import numpy as np
from typing import Dict, Optional
from training.hyperparameter_optimization import HyperparameterOptimizer, load_hyperparameters, save_hyperparameters
from training.data_preparation import check_data_quality, remove_duplicates, create_train_val_test_split
from training.config import config, create_directories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset_from_csv(csv_path: str):
    """Load dataset from CSV file"""
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if 'text' not in df.columns:
        # Try common alternative column names
        if 'body' in df.columns:
            df['text'] = df['body']
        elif 'email' in df.columns:
            df['text'] = df['email']
        elif 'content' in df.columns:
            df['text'] = df['content']
        else:
            raise ValueError("Dataset must have 'text' column (or 'body', 'email', 'content')")
    
    if 'label' not in df.columns:
        # Try common alternative column names
        if 'type' in df.columns:
            df['label'] = df['type'].map({'phishing': 1, 'legitimate': 0, 'safe': 0})
        elif 'is_phishing' in df.columns:
            df['label'] = df['is_phishing'].astype(int)
        elif 'class' in df.columns:
            df['label'] = df['class'].astype(int)
        else:
            raise ValueError("Dataset must have 'label' column (or 'type', 'is_phishing', 'class')")
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.len() > 10]  # Remove very short texts
    
    # Ensure labels are 0 or 1
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Phishing samples: {df['label'].sum()}")
    logger.info(f"Legitimate samples: {len(df) - df['label'].sum()}")
    
    return df[['text', 'label']]

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text examples"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_training_metrics(metrics: Dict, output_path: str):
    """Save training metrics to JSON file"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved training metrics to {output_path}")

def evaluate_model(trainer: Trainer, test_dataset: Dataset) -> Dict:
    """Comprehensive model evaluation"""
    logger.info("Running comprehensive evaluation...")
    
    eval_results = trainer.evaluate(test_dataset)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    # Calculate additional metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary', zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels).tolist()
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm,
        'eval_loss': float(eval_results.get('eval_loss', 0)),
        'per_class_metrics': {}
    }
    
    # Per-class metrics
    if len(cm) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        metrics['per_class_metrics'] = {
            'legitimate': {
                'precision': float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
                'recall': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            },
            'phishing': {
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            }
        }
    
    return metrics

def train_phishing_model(
    dataset_path: str = None,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./models/phishing-bert-v1",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    test_size: float = 0.15,
    val_size: float = 0.15,
    optimize: bool = False,
    hyperparameters_path: Optional[str] = None,
    use_early_stopping: bool = True,
    early_stopping_patience: int = 2
):
    """
    Train a phishing detection model
    
    Args:
        dataset_path: Path to CSV file with 'text' and 'label' columns
        model_name: Pre-trained model to fine-tune
        output_dir: Directory to save the trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        test_size: Fraction of data to use for validation
    """
    
    # Configuration
    num_labels = 2  # Binary classification: phishing vs legitimate
    create_directories()
    
    logger.info(f"Training phishing detection model: {model_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load dataset
    if dataset_path is None:
        dataset_path = config.PHISHING_DATASET_PATH
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        logger.info("Please download a dataset and save it as CSV with 'text' and 'label' columns")
        logger.info("See DATASET_GUIDE.md for dataset sources")
        return
    
    df = load_dataset_from_csv(dataset_path)
    
    # Data quality checks
    quality_report = check_data_quality(df)
    logger.info(f"Data quality report: {json.dumps(quality_report, indent=2)}")
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Hyperparameter optimization
    if optimize:
        logger.info("Running hyperparameter optimization...")
        optimizer = HyperparameterOptimizer(model_name=model_name, device=config.DEFAULT_DEVICE)
        best_params = optimizer.optimize_phishing_hyperparameters(
            dataset_path=dataset_path,
            n_trials=config.DEFAULT_N_TRIALS,
            timeout=config.OPTIMIZATION_TIMEOUT
        )
        
        # Update hyperparameters
        num_epochs = best_params.get('num_epochs', num_epochs)
        batch_size = best_params.get('batch_size', batch_size)
        learning_rate = best_params.get('learning_rate', learning_rate)
        weight_decay = best_params.get('weight_decay', weight_decay)
        warmup_steps = best_params.get('warmup_steps', warmup_steps)
        
        logger.info(f"Using optimized hyperparameters:")
        logger.info(f"  Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Save hyperparameters
        save_hyperparameters(best_params, config.PHISHING_HYPERPARAMETERS_PATH)
    elif hyperparameters_path and os.path.exists(hyperparameters_path):
        logger.info(f"Loading hyperparameters from {hyperparameters_path}")
        best_params = load_hyperparameters(hyperparameters_path)
        num_epochs = best_params.get('num_epochs', num_epochs)
        batch_size = best_params.get('batch_size', batch_size)
        learning_rate = best_params.get('learning_rate', learning_rate)
        weight_decay = best_params.get('weight_decay', weight_decay)
        warmup_steps = best_params.get('warmup_steps', warmup_steps)
    
    # Split into train, validation, and test
    train_df, val_df, test_df = create_train_val_test_split(
        df,
        train_size=1 - test_size - val_size,
        val_size=val_size,
        test_size=test_size,
        stratify=True,
        random_state=42
    )
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    # Remove only text column, keep label
    columns_to_remove = [col for col in train_dataset.column_names if col != 'label']
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=columns_to_remove
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=columns_to_remove
    )
    
    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        fp16=config.MIXED_PRECISION and config.USE_GPU,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Callbacks
    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_metrics = evaluate_model(trainer, val_dataset)
    
    # Evaluate on test set
    test_dataset = Dataset.from_pandas(test_df)
    test_columns_to_remove = [col for col in test_dataset.column_names if col != 'label']
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=test_columns_to_remove
    )
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(trainer, test_dataset)
    
    # Save metrics
    all_metrics = {
        'model_name': model_name,
        'training_loss': float(train_result.training_loss),
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'hyperparameters': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
        }
    }
    
    metrics_path = os.path.join(config.METRICS_DIR, "phishing_training_metrics.json")
    save_training_metrics(all_metrics, metrics_path)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    logger.info(f"Test - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    logger.info("=" * 60)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train phishing detection model")
    parser.add_argument("--dataset", type=str, help="Path to CSV dataset file")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Model name")
    parser.add_argument("--output", type=str, default="./models/phishing-bert-v1", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--hyperparameters", type=str, help="Path to hyperparameters JSON file")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    
    args = parser.parse_args()
    
    train_phishing_model(
        dataset_path=args.dataset,
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimize=args.optimize,
        hyperparameters_path=args.hyperparameters,
        use_early_stopping=not args.no_early_stopping
    )
