"""
Hyperparameter optimization module using Optuna
Automatically finds best hyperparameters for maximum accuracy
"""
import os
import sys
import json
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
from optuna.trial import TrialState
import pandas as pd
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
import torch
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Hyperparameter optimizer using Optuna"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.best_params = None
        self.best_score = None
    
    def optimize_phishing_hyperparameters(
        self,
        dataset_path: str,
        n_trials: int = 20,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Optimize hyperparameters for phishing detection model
        
        Args:
            dataset_path: Path to CSV dataset
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
            
        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization for phishing model")
        logger.info(f"Model: {self.model_name}, Trials: {n_trials}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Prepare data
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'label' columns")
        
        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].astype(int)
        df = df[df['label'].isin([0, 1])]
        
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=42
        )
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512
            )
        
        # Remove only text column, keep label
        columns_to_remove = [col for col in train_dataset.column_names if col != 'label']
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
        
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        # Define objective function
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
            num_epochs = trial.suggest_int("num_epochs", 3, 7)
            weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
            warmup_steps = trial.suggest_int("warmup_steps", 100, 1000, step=100)
            
            # Create model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2
            )
            model.to(self.device)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./models/temp_optuna_{trial.number}",
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                logging_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to="none",  # Disable wandb/tensorboard
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = predictions.argmax(axis=-1)
                accuracy = (predictions == labels).mean()
                return {"accuracy": accuracy}
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            # Train
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            accuracy = eval_results.get("eval_accuracy", 0.0)
            
            # Cleanup
            import shutil
            if os.path.exists(f"./models/temp_optuna_{trial.number}"):
                shutil.rmtree(f"./models/temp_optuna_{trial.number}")
            
            return accuracy
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name="phishing_hyperparameter_optimization"
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"✓ Optimization completed")
        logger.info(f"✓ Best accuracy: {self.best_score:.4f}")
        logger.info(f"✓ Best parameters: {self.best_params}")
        
        return self.best_params
    
    def optimize_ai_detector_hyperparameters(
        self,
        dataset_path: Optional[str] = None,
        use_huggingface: bool = True,
        huggingface_dataset: str = "shahxeebhassan/human_vs_ai_sentences",
        n_trials: int = 20,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Optimize hyperparameters for AI detection model
        
        Args:
            dataset_path: Path to CSV dataset (if not using Hugging Face)
            use_huggingface: Whether to use Hugging Face dataset
            huggingface_dataset: Hugging Face dataset name
            n_trials: Number of optimization trials
            timeout: Maximum time in seconds for optimization
            
        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization for AI detector")
        logger.info(f"Model: {self.model_name}, Trials: {n_trials}")
        
        # Load dataset
        if use_huggingface:
            from datasets import load_dataset
            dataset = load_dataset(huggingface_dataset)
            df = pd.DataFrame(dataset['train'])
            
            if 'sentence' in df.columns and 'text' not in df.columns:
                df['text'] = df['sentence']
            if 'generated' in df.columns and 'label' not in df.columns:
                df['label'] = df['generated'].astype(int)
        else:
            if dataset_path is None:
                raise ValueError("dataset_path required when use_huggingface=False")
            df = pd.read_csv(dataset_path)
        
        # Prepare data
        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].astype(int)
        df = df[df['label'].isin([0, 1])]
        
        # Use subset for faster optimization (optional)
        if len(df) > 10000:
            logger.info(f"Using subset of {len(df)} samples for faster optimization")
            df = df.sample(n=10000, random_state=42)
        
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=42
        )
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512
            )
        
        # Remove only text column, keep label
        columns_to_remove = [col for col in train_dataset.column_names if col != 'label']
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
        
        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        
        # Define objective function
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
            num_epochs = trial.suggest_int("num_epochs", 3, 7)
            weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
            warmup_steps = trial.suggest_int("warmup_steps", 100, 1000, step=100)
            
            # Create model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2
            )
            model.to(self.device)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./models/temp_optuna_ai_{trial.number}",
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                logging_steps=50,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to="none",
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = predictions.argmax(axis=-1)
                accuracy = (predictions == labels).mean()
                return {"accuracy": accuracy}
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            # Train
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            accuracy = eval_results.get("eval_accuracy", 0.0)
            
            # Cleanup
            import shutil
            if os.path.exists(f"./models/temp_optuna_ai_{trial.number}"):
                shutil.rmtree(f"./models/temp_optuna_ai_{trial.number}")
            
            return accuracy
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name="ai_detector_hyperparameter_optimization"
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"✓ Optimization completed")
        logger.info(f"✓ Best accuracy: {self.best_score:.4f}")
        logger.info(f"✓ Best parameters: {self.best_params}")
        
        return self.best_params

def save_hyperparameters(params: Dict, output_path: str):
    """Save hyperparameters to JSON file"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters to {output_path}")

def load_hyperparameters(input_path: str) -> Dict:
    """Load hyperparameters from JSON file"""
    with open(input_path, 'r') as f:
        params = json.load(f)
    logger.info(f"Loaded hyperparameters from {input_path}")
    return params

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument("--phishing", action="store_true", help="Optimize phishing model")
    parser.add_argument("--ai", action="store_true", help="Optimize AI detector")
    parser.add_argument("--dataset", type=str, help="Path to dataset CSV")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Base model name")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    
    args = parser.parse_args()
    
    optimizer = HyperparameterOptimizer(model_name=args.model)
    
    if args.phishing:
        if not args.dataset:
            args.dataset = "data/raw/phishing_emails.csv"
        params = optimizer.optimize_phishing_hyperparameters(
            dataset_path=args.dataset,
            n_trials=args.trials,
            timeout=args.timeout
        )
        save_hyperparameters(params, "models/hyperparameters/phishing_best_params.json")
    
    elif args.ai:
        params = optimizer.optimize_ai_detector_hyperparameters(
            dataset_path=args.dataset,
            use_huggingface=not args.dataset,
            n_trials=args.trials,
            timeout=args.timeout
        )
        save_hyperparameters(params, "models/hyperparameters/ai_detector_best_params.json")
    
    else:
        logger.info("Usage:")
        logger.info("  python hyperparameter_optimization.py --phishing --dataset data/raw/phishing_emails.csv")
        logger.info("  python hyperparameter_optimization.py --ai")
