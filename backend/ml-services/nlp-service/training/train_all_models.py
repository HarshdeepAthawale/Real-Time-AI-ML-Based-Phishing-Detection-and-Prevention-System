"""
Master training script - Orchestrates complete training pipeline
Downloads datasets, optimizes hyperparameters, trains models, and evaluates
"""
import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, Optional
from training.download_datasets import download_all_datasets, download_ai_detection_dataset, download_utwente_phishing_dataset
from training.hyperparameter_optimization import HyperparameterOptimizer, save_hyperparameters
from training.train_phishing_model import train_phishing_model
from training.train_ai_detector import train_ai_detector
from training.evaluate_models import evaluate_model_comprehensive, generate_evaluation_report
from training.config import config, create_directories
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

# Create metrics directory if it doesn't exist
os.makedirs('data/metrics', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data/metrics/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_datasets_exist() -> Dict[str, bool]:
    """Check if datasets exist"""
    datasets = {
        'phishing': os.path.exists(config.PHISHING_DATASET_PATH),
        'ai_detection': os.path.exists(config.AI_DETECTION_DATASET_PATH)
    }
    return datasets

def download_datasets_if_needed(force_download: bool = False) -> Dict[str, bool]:
    """Download datasets if they don't exist"""
    logger.info("Checking datasets...")
    datasets_exist = check_datasets_exist()
    
    results = {
        'phishing': False,
        'ai_detection': False
    }
    
    if not datasets_exist['phishing'] or force_download:
        logger.info("Downloading phishing dataset...")
        result = download_utwente_phishing_dataset()
        results['phishing'] = result is not None
    else:
        logger.info("Phishing dataset already exists")
        results['phishing'] = True
    
    if not datasets_exist['ai_detection'] or force_download:
        logger.info("Downloading AI detection dataset...")
        result = download_ai_detection_dataset()
        results['ai_detection'] = result is not None
    else:
        logger.info("AI detection dataset already exists")
        results['ai_detection'] = True
    
    return results

def optimize_hyperparameters(
    optimize_phishing: bool = True,
    optimize_ai: bool = True,
    n_trials: int = None
) -> Dict[str, Dict]:
    """Run hyperparameter optimization for both models"""
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 60)
    
    best_params = {}
    n_trials = n_trials or config.DEFAULT_N_TRIALS
    
    if optimize_phishing:
        logger.info("\n[1/2] Optimizing phishing model hyperparameters...")
        optimizer = HyperparameterOptimizer(
            model_name=config.PHISHING_MODELS[0],
            device=config.DEFAULT_DEVICE
        )
        phishing_params = optimizer.optimize_phishing_hyperparameters(
            dataset_path=config.PHISHING_DATASET_PATH,
            n_trials=n_trials,
            timeout=config.OPTIMIZATION_TIMEOUT
        )
        best_params['phishing'] = phishing_params
        save_hyperparameters(phishing_params, config.PHISHING_HYPERPARAMETERS_PATH)
    
    if optimize_ai:
        logger.info("\n[2/2] Optimizing AI detector hyperparameters...")
        optimizer = HyperparameterOptimizer(
            model_name=config.AI_DETECTOR_MODELS[0],
            device=config.DEFAULT_DEVICE
        )
        ai_params = optimizer.optimize_ai_detector_hyperparameters(
            use_huggingface=True,
            huggingface_dataset="shahxeebhassan/human_vs_ai_sentences",
            n_trials=n_trials,
            timeout=config.OPTIMIZATION_TIMEOUT
        )
        best_params['ai'] = ai_params
        save_hyperparameters(ai_params, config.AI_DETECTOR_HYPERPARAMETERS_PATH)
    
    return best_params

def train_all_models(
    use_optimized_params: bool = True,
    train_phishing: bool = True,
    train_ai: bool = True
) -> Dict[str, bool]:
    """Train both models"""
    logger.info("=" * 60)
    logger.info("MODEL TRAINING")
    logger.info("=" * 60)
    
    results = {
        'phishing': False,
        'ai': False
    }
    
    if train_phishing:
        logger.info("\n[1/2] Training phishing detection model...")
        try:
            train_phishing_model(
                dataset_path=config.PHISHING_DATASET_PATH,
                model_name=config.PHISHING_MODELS[0],
                output_dir=config.PHISHING_MODEL_PATH,
                optimize=False,  # Already optimized if use_optimized_params
                hyperparameters_path=config.PHISHING_HYPERPARAMETERS_PATH if use_optimized_params else None
            )
            results['phishing'] = True
        except Exception as e:
            logger.error(f"Failed to train phishing model: {e}", exc_info=True)
    
    if train_ai:
        logger.info("\n[2/2] Training AI detector model...")
        try:
            train_ai_detector(
                use_huggingface=True,
                huggingface_dataset="shahxeebhassan/human_vs_ai_sentences",
                model_name=config.AI_DETECTOR_MODELS[0],
                output_dir=config.AI_DETECTOR_MODEL_PATH,
                optimize=False,  # Already optimized if use_optimized_params
                hyperparameters_path=config.AI_DETECTOR_HYPERPARAMETERS_PATH if use_optimized_params else None
            )
            results['ai'] = True
        except Exception as e:
            logger.error(f"Failed to train AI detector: {e}", exc_info=True)
    
    return results

def evaluate_all_models() -> Dict[str, Dict]:
    """Evaluate all trained models"""
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    results = {}
    
    # Evaluate phishing model
    if os.path.exists(config.PHISHING_MODEL_PATH):
        logger.info("\n[1/2] Evaluating phishing model...")
        try:
            # Load test data
            df = pd.read_csv(config.PHISHING_DATASET_PATH)
            from training.data_preparation import create_train_val_test_split
            _, _, test_df = create_train_val_test_split(df, random_state=42)
            
            test_dataset = Dataset.from_pandas(test_df)
            tokenizer = AutoTokenizer.from_pretrained(config.PHISHING_MODELS[0])
            
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=512
                )
            
            test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
            test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            
            metrics = evaluate_model_comprehensive(
                model_path=config.PHISHING_MODEL_PATH,
                test_dataset=test_dataset,
                model_name=config.PHISHING_MODELS[0],
                device=config.DEFAULT_DEVICE
            )
            results['phishing'] = metrics
            
            generate_evaluation_report(
                metrics, "phishing_model", config.METRICS_DIR,
                class_names=['Legitimate', 'Phishing']
            )
        except Exception as e:
            logger.error(f"Failed to evaluate phishing model: {e}", exc_info=True)
    
    # Evaluate AI detector
    if os.path.exists(config.AI_DETECTOR_MODEL_PATH):
        logger.info("\n[2/2] Evaluating AI detector...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("shahxeebhassan/human_vs_ai_sentences")
            df = pd.DataFrame(dataset['train'])
            
            if 'sentence' in df.columns:
                df['text'] = df['sentence']
            if 'generated' in df.columns:
                df['label'] = df['generated']
            
            from training.data_preparation import create_train_val_test_split
            _, _, test_df = create_train_val_test_split(df, random_state=42)
            
            test_dataset = Dataset.from_pandas(test_df)
            tokenizer = AutoTokenizer.from_pretrained(config.AI_DETECTOR_MODELS[0])
            
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    truncation=True,
                    padding='max_length',
                    max_length=512
                )
            
            test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
            test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            
            metrics = evaluate_model_comprehensive(
                model_path=config.AI_DETECTOR_MODEL_PATH,
                test_dataset=test_dataset,
                model_name=config.AI_DETECTOR_MODELS[0],
                device=config.DEFAULT_DEVICE
            )
            results['ai'] = metrics
            
            generate_evaluation_report(
                metrics, "ai_detector", config.METRICS_DIR,
                class_names=['Human-written', 'AI-generated']
            )
        except Exception as e:
            logger.error(f"Failed to evaluate AI detector: {e}", exc_info=True)
    
    return results

def generate_training_report(results: Dict) -> str:
    """Generate final training report"""
    report_path = os.path.join(config.METRICS_DIR, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    
    with open(report_path, 'w') as f:
        f.write("# Training Pipeline Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Datasets downloaded: {'✓' if results.get('datasets', {}).get('phishing') and results.get('datasets', {}).get('ai_detection') else '✗'}\n")
        f.write(f"- Hyperparameters optimized: {'✓' if results.get('optimization') else '✗'}\n")
        f.write(f"- Models trained: {'✓' if results.get('training', {}).get('phishing') and results.get('training', {}).get('ai') else '✗'}\n")
        f.write(f"- Models evaluated: {'✓' if results.get('evaluation') else '✗'}\n\n")
        
        if 'evaluation' in results:
            f.write("## Evaluation Results\n\n")
            for model_name, metrics in results['evaluation'].items():
                f.write(f"### {model_name}\n\n")
                f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"- F1-Score: {metrics['f1_macro']:.4f}\n")
                f.write(f"- ROC AUC: {metrics['roc_auc']:.4f}\n\n")
    
    logger.info(f"Saved training report to {report_path}")
    return report_path

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Complete training pipeline")
    parser.add_argument("--download", action="store_true", help="Download datasets")
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--force-download", action="store_true", help="Force re-download datasets")
    parser.add_argument("--trials", type=int, help="Number of optimization trials")
    parser.add_argument("--phishing-only", action="store_true", help="Only train phishing model")
    parser.add_argument("--ai-only", action="store_true", help="Only train AI detector")
    
    args = parser.parse_args()
    
    # If --all, set all flags
    if args.all:
        args.download = True
        args.optimize = True
        args.train = True
        args.evaluate = True
    
    # Create directories
    create_directories()
    
    results = {}
    
    # Step 1: Download datasets
    if args.download:
        logger.info("=" * 60)
        logger.info("STEP 1: DOWNLOADING DATASETS")
        logger.info("=" * 60)
        results['datasets'] = download_datasets_if_needed(force_download=args.force_download)
    
    # Step 2: Optimize hyperparameters
    if args.optimize:
        logger.info("=" * 60)
        logger.info("STEP 2: HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 60)
        results['optimization'] = optimize_hyperparameters(
            optimize_phishing=not args.ai_only,
            optimize_ai=not args.phishing_only,
            n_trials=args.trials
        )
    
    # Step 3: Train models
    if args.train:
        logger.info("=" * 60)
        logger.info("STEP 3: TRAINING MODELS")
        logger.info("=" * 60)
        results['training'] = train_all_models(
            use_optimized_params=args.optimize,
            train_phishing=not args.ai_only,
            train_ai=not args.phishing_only
        )
    
    # Step 4: Evaluate models
    if args.evaluate:
        logger.info("=" * 60)
        logger.info("STEP 4: EVALUATING MODELS")
        logger.info("=" * 60)
        results['evaluation'] = evaluate_all_models()
    
    # Generate final report
    if results:
        report_path = generate_training_report(results)
        logger.info(f"\n{'=' * 60}")
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info(f"{'=' * 60}")
        logger.info(f"Report saved to: {report_path}")
    else:
        logger.info("No steps executed. Use --all or specify individual steps.")

if __name__ == "__main__":
    main()
