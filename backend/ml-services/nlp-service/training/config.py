"""
Centralized configuration for training
"""
import os
import sys
from typing import Dict, List
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class TrainingConfig:
    """Training configuration"""
    
    # Model architectures
    PHISHING_MODELS: List[str] = None
    AI_DETECTOR_MODELS: List[str] = None
    
    # Hyperparameter ranges
    LEARNING_RATE_RANGE: tuple = (1e-5, 5e-5)
    BATCH_SIZE_OPTIONS: List[int] = None
    EPOCHS_RANGE: tuple = (3, 7)
    WEIGHT_DECAY_RANGE: tuple = (0.01, 0.1)
    WARMUP_STEPS_OPTIONS: List[int] = None
    
    # Training paths
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    MODELS_DIR: str = "models"
    METRICS_DIR: str = "data/metrics"
    HYPERPARAMETERS_DIR: str = "models/hyperparameters"
    
    # Dataset paths
    PHISHING_DATASET_PATH: str = "data/raw/phishing_emails.csv"
    AI_DETECTION_DATASET_PATH: str = "data/raw/ai_detection.csv"
    
    # Model output paths
    PHISHING_MODEL_PATH: str = "models/phishing-bert-v1"
    AI_DETECTOR_MODEL_PATH: str = "models/ai-detector-v1"
    
    # Hyperparameter files
    PHISHING_HYPERPARAMETERS_PATH: str = "models/hyperparameters/phishing_best_params.json"
    AI_DETECTOR_HYPERPARAMETERS_PATH: str = "models/hyperparameters/ai_detector_best_params.json"
    
    # Training settings
    DEFAULT_DEVICE: str = "cpu"
    USE_GPU: bool = False
    MIXED_PRECISION: bool = False
    EARLY_STOPPING_PATIENCE: int = 2
    
    # Optimization settings
    DEFAULT_N_TRIALS: int = 20
    OPTIMIZATION_TIMEOUT: int = None  # None = no timeout
    
    # Evaluation metrics
    EVALUATION_METRICS: List[str] = None
    
    # Data split
    TRAIN_SIZE: float = 0.7
    VAL_SIZE: float = 0.15
    TEST_SIZE: float = 0.15
    
    def __post_init__(self):
        """Initialize default values"""
        if self.PHISHING_MODELS is None:
            self.PHISHING_MODELS = [
                "distilbert-base-uncased",
                "roberta-base",
                "bert-base-uncased"
            ]
        
        if self.AI_DETECTOR_MODELS is None:
            self.AI_DETECTOR_MODELS = [
                "roberta-base",
                "distilbert-base-uncased",
                "bert-base-uncased"
            ]
        
        if self.BATCH_SIZE_OPTIONS is None:
            self.BATCH_SIZE_OPTIONS = [8, 16, 32]
        
        if self.WARMUP_STEPS_OPTIONS is None:
            self.WARMUP_STEPS_OPTIONS = [100, 500, 1000]
        
        if self.EVALUATION_METRICS is None:
            self.EVALUATION_METRICS = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "confusion_matrix"
            ]
        
        # Check for GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.USE_GPU = True
                self.DEFAULT_DEVICE = "cuda"
        except ImportError:
            pass

# Global configuration instance
config = TrainingConfig()

# Environment variable overrides
config.DEFAULT_DEVICE = os.getenv("DEVICE", config.DEFAULT_DEVICE)
config.USE_GPU = os.getenv("USE_GPU", "false").lower() == "true" or config.USE_GPU
config.MIXED_PRECISION = os.getenv("MIXED_PRECISION", "false").lower() == "true"
config.DEFAULT_N_TRIALS = int(os.getenv("N_TRIALS", config.DEFAULT_N_TRIALS))

# Dataset paths from environment
config.PHISHING_DATASET_PATH = os.getenv("PHISHING_DATASET_PATH", config.PHISHING_DATASET_PATH)
config.AI_DETECTION_DATASET_PATH = os.getenv("AI_DETECTION_DATASET_PATH", config.AI_DETECTION_DATASET_PATH)

# Model paths from environment
config.PHISHING_MODEL_PATH = os.getenv("PHISHING_MODEL_PATH", config.PHISHING_MODEL_PATH)
config.AI_DETECTOR_MODEL_PATH = os.getenv("AI_DETECTOR_MODEL_PATH", config.AI_DETECTOR_MODEL_PATH)

def get_config() -> TrainingConfig:
    """Get training configuration"""
    return config

def create_directories():
    """Create necessary directories"""
    dirs = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.METRICS_DIR,
        config.HYPERPARAMETERS_DIR,
        config.PHISHING_MODEL_PATH,
        config.AI_DETECTOR_MODEL_PATH,
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
