"""
Training script for AI-generated content detector
Fine-tune a model to distinguish between AI-generated and human-written text
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
from datasets import Dataset, load_dataset
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
            'human_written': {
                'precision': float(tn / (tn + fn)) if (tn + fn) > 0 else 0,
                'recall': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
            },
            'ai_generated': {
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            }
        }
    
    return metrics

def sample_dataset_stratified(df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """
    Sample dataset using stratified sampling to maintain class balance.
    
    Args:
        df: DataFrame with 'label' column
        max_samples: Maximum number of samples to return
        
    Returns:
        Sampled DataFrame with balanced class distribution
    """
    if max_samples is None or len(df) <= max_samples:
        return df
    
    original_size = len(df)
    num_classes = df['label'].nunique()
    
    # Calculate samples per class (balanced)
    samples_per_class = max_samples // num_classes
    
    # Sample from each class
    sampled_dfs = []
    sampled_indices = set()
    for label in df['label'].unique():
        class_df = df[df['label'] == label]
        if len(class_df) >= samples_per_class:
            sampled_class = class_df.sample(n=samples_per_class, random_state=42)
        else:
            # If class has fewer samples, take all
            sampled_class = class_df
        sampled_dfs.append(sampled_class)
        sampled_indices.update(sampled_class.index)
    
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we need more samples to reach max_samples, randomly sample from remaining
    if len(sampled_df) < max_samples:
        remaining = df[~df.index.isin(sampled_indices)]
        needed = max_samples - len(sampled_df)
        if len(remaining) > 0:
            additional = remaining.sample(n=min(needed, len(remaining)), random_state=42)
            sampled_df = pd.concat([sampled_df, additional], ignore_index=True)
    
    # Shuffle the final dataset
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Sampled {len(sampled_df)} samples from {original_size} (stratified)")
    logger.info(f"Class distribution after sampling:")
    for label in sorted(sampled_df['label'].unique()):
        count = (sampled_df['label'] == label).sum()
        logger.info(f"  Label {label}: {count} samples ({count/len(sampled_df)*100:.1f}%)")
    
    return sampled_df

def load_dataset_from_csv(csv_path: str, max_samples: Optional[int] = None):
    """
    Load dataset from CSV file
    
    Args:
        csv_path: Path to CSV file
        max_samples: Maximum number of samples to load (uses stratified sampling)
    
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if 'text' not in df.columns:
        if 'sentence' in df.columns:
            df['text'] = df['sentence']
        elif 'content' in df.columns:
            df['text'] = df['content']
        else:
            raise ValueError("Dataset must have 'text' column")
    
    if 'label' not in df.columns:
        if 'is_ai' in df.columns:
            df['label'] = df['is_ai'].astype(int)
        elif 'generated' in df.columns:
            df['label'] = df['generated'].astype(int)
        else:
            raise ValueError("Dataset must have 'label' column")
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.len() > 10]
    
    # Ensure labels are 0 or 1
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"AI-generated samples: {df['label'].sum()}")
    logger.info(f"Human-written samples: {len(df) - df['label'].sum()}")
    
    df = df[['text', 'label']]
    
    # Apply stratified sampling if max_samples is specified
    if max_samples is not None:
        df = sample_dataset_stratified(df, max_samples)
    
    return df

def load_dataset_from_huggingface(dataset_name: str = "shahxeebhassan/human_vs_ai_sentences", max_samples: Optional[int] = None):
    """
    Load dataset from Hugging Face
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        max_samples: Maximum number of samples to load (uses stratified sampling)
    
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    logger.info(f"Loading dataset from Hugging Face: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name)
        
        # Convert to pandas DataFrame
        if 'train' in dataset:
            df = pd.DataFrame(dataset['train'])
        else:
            df = pd.DataFrame(dataset)
        
        # Ensure correct column names
        if 'text' not in df.columns and 'sentence' in df.columns:
            df['text'] = df['sentence']
        
        if 'label' not in df.columns and 'generated' in df.columns:
            df['label'] = df['generated'].astype(int)
        
        logger.info(f"Loaded {len(df)} samples from Hugging Face")
        df = df[['text', 'label']]
        
        # Apply stratified sampling if max_samples is specified
        if max_samples is not None:
            df = sample_dataset_stratified(df, max_samples)
        
        return df
    except Exception as e:
        logger.error(f"Failed to load from Hugging Face: {e}")
        raise

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text examples"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

def train_ai_detector(
    dataset_path: str = None,
    use_huggingface: bool = True,
    huggingface_dataset: str = "shahxeebhassan/human_vs_ai_sentences",
    model_name: str = "roberta-base",
    output_dir: str = "./models/ai-detector-v1",
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
    early_stopping_patience: int = 2,
    max_samples: Optional[int] = None
):
    """
    Train an AI-generated content detection model
    
    Args:
        dataset_path: Path to CSV file (if not using Hugging Face)
        use_huggingface: Whether to use Hugging Face dataset
        huggingface_dataset: Hugging Face dataset name
        model_name: Pre-trained model to fine-tune
        output_dir: Directory to save the trained model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        test_size: Fraction of data to use for test set
        val_size: Fraction of data to use for validation set
        optimize: Whether to run hyperparameter optimization
        hyperparameters_path: Path to saved hyperparameters JSON file
        use_early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        max_samples: Maximum number of samples to use (for faster training, uses stratified sampling)
    """
    
    # Configuration
    num_labels = 2  # Binary classification: AI-generated vs human-written
    create_directories()
    
    logger.info(f"Training AI detector model: {model_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load dataset
    if use_huggingface:
        try:
            df = load_dataset_from_huggingface(huggingface_dataset, max_samples=max_samples)
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            logger.info("Falling back to CSV dataset")
            use_huggingface = False
    
    if not use_huggingface:
        if dataset_path is None:
            dataset_path = config.AI_DETECTION_DATASET_PATH
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found at {dataset_path}")
            logger.info("Please download a dataset:")
            logger.info("1. From Hugging Face: datasets.load_dataset('shahxeebhassan/human_vs_ai_sentences')")
            logger.info("2. Or save CSV with 'text' and 'label' columns")
            logger.info("See DATASET_GUIDE.md for more options")
            return
        
        df = load_dataset_from_csv(dataset_path, max_samples=max_samples)
    
    # Data quality checks
    quality_report = check_data_quality(df)
    logger.info(f"Data quality report: {json.dumps(quality_report, indent=2)}")
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Hyperparameter optimization
    if optimize:
        logger.info("Running hyperparameter optimization...")
        optimizer = HyperparameterOptimizer(model_name=model_name, device=config.DEFAULT_DEVICE)
        best_params = optimizer.optimize_ai_detector_hyperparameters(
            dataset_path=dataset_path if not use_huggingface else None,
            use_huggingface=use_huggingface,
            huggingface_dataset=huggingface_dataset,
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
        save_hyperparameters(best_params, config.AI_DETECTOR_HYPERPARAMETERS_PATH)
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
    
    # Calculate and log training time estimation
    steps_per_epoch = len(train_df) // batch_size
    total_steps = steps_per_epoch * num_epochs
    # Estimate ~0.5 seconds per step on CPU (may vary)
    estimated_seconds = total_steps * 0.5
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_minutes / 60
    
    if estimated_hours >= 1:
        logger.info(f"Estimated training time: ~{estimated_hours:.1f} hours ({estimated_minutes:.1f} minutes)")
    else:
        logger.info(f"Estimated training time: ~{estimated_minutes:.1f} minutes")
    logger.info(f"Total training steps: {total_steps} ({steps_per_epoch} steps/epoch × {num_epochs} epochs)")
    
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
    
    metrics_path = os.path.join(config.METRICS_DIR, "ai_detector_training_metrics.json")
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
    
    parser = argparse.ArgumentParser(description="Train AI detection model")
    parser.add_argument("--dataset", type=str, help="Path to CSV dataset file")
    parser.add_argument("--huggingface", action="store_true", help="Use Hugging Face dataset")
    parser.add_argument("--hf-dataset", type=str, default="shahxeebhassan/human_vs_ai_sentences", help="Hugging Face dataset name")
    parser.add_argument("--model", type=str, default="roberta-base", help="Model name")
    parser.add_argument("--output", type=str, default="./models/ai-detector-v1", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--hyperparameters", type=str, help="Path to hyperparameters JSON file")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="Maximum number of samples to use (for faster training, uses stratified sampling)")
    
    args = parser.parse_args()
    
    train_ai_detector(
        dataset_path=args.dataset,
        use_huggingface=args.huggingface,
        huggingface_dataset=args.hf_dataset,
        model_name=args.model,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        optimize=args.optimize,
        hyperparameters_path=args.hyperparameters,
        use_early_stopping=not args.no_early_stopping,
        max_samples=args.max_samples
    )
