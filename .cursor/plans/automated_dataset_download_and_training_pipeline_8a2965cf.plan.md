---
name: Automated Dataset Download and Training Pipeline
overview: Create an automated pipeline to download both datasets (UTwente phishing and Hugging Face AI detection), preprocess them, and train both models with hyperparameter optimization for best accuracy.
todos: []
---

# Automated Dataset Download and Training Pipeline

## Overview

Automate downloading of UTwente phishing dataset (~200 KB) and Hugging Face AI detection dataset, then train both models with hyperparameter optimization for maximum accuracy.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Automated Training Pipeline                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Download Phase                                      │
│     ├─ UTwente Dataset (Zenodo API)                     │
│     └─ Hugging Face AI Dataset                           │
│                                                          │
│  2. Preprocessing Phase                                 │
│     ├─ Data validation & cleaning                       │
│     ├─ Train/Val/Test split                             │
│     └─ Format conversion                                │
│                                                          │
│  3. Training Phase                                      │
│     ├─ Hyperparameter optimization                      │
│     ├─ Model training with best params                  │
│     └─ Model evaluation                                 │
│                                                          │
│  4. Model Saving                                        │
│     └─ Save to models/ directory                        │
└─────────────────────────────────────────────────────────┘
```

## Implementation Plan

### 1. Enhanced Download Script

**File**: `backend/ml-services/nlp-service/training/download_datasets.py`

**Changes**:

- Add `download_utwente_phishing_dataset()` function
  - Use Zenodo API: `https://zenodo.org/api/records/13474746/files`
  - Download CSV: `Phishing_validation_emails.csv`
  - Handle column mapping (Safe/Phishing → 0/1)
  - Save to `data/raw/phishing_emails.csv`
- Enhance `download_ai_detection_dataset()` with progress tracking
- Add `download_all_datasets()` function to download both
- Add data validation after download
- Handle download errors gracefully with retries

**Key Features**:

- Direct Zenodo API download (no manual steps)
- Automatic column name mapping
- Data validation (check for required columns, data quality)
- Progress indicators
- Error handling with retry logic

### 2. Hyperparameter Optimization Module

**File**: `backend/ml-services/nlp-service/training/hyperparameter_optimization.py` (NEW)

**Purpose**: Automatically find best hyperparameters for maximum accuracy

**Implementation**:

- Use Optuna or scikit-learn's GridSearchCV
- Search space:
  - Learning rate: [1e-5, 2e-5, 3e-5, 5e-5]
  - Batch size: [8, 16, 32] (based on available memory)
  - Epochs: [3, 5, 7] (with early stopping)
  - Weight decay: [0.01, 0.1]
  - Warmup steps: [100, 500, 1000]
- Objective: Maximize validation accuracy
- Use cross-validation or hold-out validation set
- Save best hyperparameters to JSON file

**Functions**:

- `optimize_phishing_hyperparameters(dataset_path) -> dict`
- `optimize_ai_detector_hyperparameters(dataset_path) -> dict`
- `save_hyperparameters(params, output_path)`
- `load_hyperparameters(input_path) -> dict`

### 3. Enhanced Training Scripts

#### 3.1 Phishing Model Training

**File**: `backend/ml-services/nlp-service/training/train_phishing_model.py`

**Enhancements**:

- Add hyperparameter optimization integration
- Add early stopping callback
- Add learning rate scheduling
- Add model checkpointing (save best model)
- Add comprehensive evaluation metrics (accuracy, precision, recall, F1)
- Add training progress visualization
- Add support for different model architectures (distilbert, roberta, bert)
- Add data augmentation option (optional)
- Save training history and metrics

**New Features**:

- `train_with_optimization()` - Use optimized hyperparameters
- `evaluate_model()` - Comprehensive evaluation
- `save_training_metrics()` - Save metrics to JSON
- Support for mixed precision training (if GPU available)

#### 3.2 AI Detector Training

**File**: `backend/ml-services/nlp-service/training/train_ai_detector.py`

**Same enhancements as phishing model**:

- Hyperparameter optimization
- Early stopping
- Learning rate scheduling
- Comprehensive evaluation
- Model checkpointing

### 4. Data Preprocessing Enhancement

**File**: `backend/ml-services/nlp-service/training/data_preparation.py`

**Enhancements**:

- Add `preprocess_utwente_dataset()` function
  - Handle Safe/Phishing label mapping
  - Combine subject + body if separate columns
  - Text cleaning and normalization
- Add `create_train_val_test_split()` with stratification
- Add data quality checks:
  - Check for class imbalance
  - Check for duplicate samples
  - Check for empty/malformed data
- Add data augmentation functions (optional):
  - Synonym replacement
  - Random word deletion
  - Text paraphrasing

### 5. Master Training Script

**File**: `backend/ml-services/nlp-service/training/train_all_models.py` (NEW)

**Purpose**: Orchestrate complete training pipeline

**Workflow**:

1. Check if datasets exist, download if missing
2. Preprocess and validate datasets
3. Run hyperparameter optimization for both models
4. Train both models with best hyperparameters
5. Evaluate models
6. Save models and metrics
7. Generate training report

**Features**:

- Command-line interface with options
- Progress tracking
- Error handling and recovery
- Logging to file
- Summary report generation

### 6. Configuration File

**File**: `backend/ml-services/nlp-service/training/config.py` (NEW)

**Purpose**: Centralized configuration for training

**Contents**:

- Default hyperparameter ranges
- Model architecture choices
- Training paths and directories
- Evaluation metrics
- Training flags (use GPU, mixed precision, etc.)

### 7. Evaluation and Reporting

**File**: `backend/ml-services/nlp-service/training/evaluate_models.py` (NEW)

**Purpose**: Comprehensive model evaluation

**Features**:

- Calculate accuracy, precision, recall, F1-score
- Confusion matrix generation
- ROC curve and AUC
- Per-class performance metrics
- Generate evaluation report (JSON + markdown)
- Compare model performance

### 8. Requirements Update

**File**: `backend/ml-services/nlp-service/requirements.txt`

**Add**:

- `optuna>=3.0.0` - Hyperparameter optimization
- `requests>=2.31.0` - For Zenodo API
- `matplotlib>=3.7.0` - For visualization
- `seaborn>=0.12.0` - For plots

## File Structure

```
backend/ml-services/nlp-service/
├── training/
│   ├── download_datasets.py          # Enhanced with UTwente download
│   ├── data_preparation.py           # Enhanced preprocessing
│   ├── hyperparameter_optimization.py # NEW - Optuna-based optimization
│   ├── train_phishing_model.py       # Enhanced with optimization
│   ├── train_ai_detector.py           # Enhanced with optimization
│   ├── train_all_models.py            # NEW - Master script
│   ├── evaluate_models.py            # NEW - Evaluation
│   ├── config.py                      # NEW - Configuration
│   └── __init__.py
├── data/
│   ├── raw/                           # Downloaded datasets
│   ├── processed/                     # Preprocessed datasets
│   └── metrics/                       # Training metrics
└── models/
    ├── phishing-bert-v1/              # Trained phishing model
    ├── ai-detector-v1/                # Trained AI detector
    └── hyperparameters/               # Saved hyperparameters
```

## Usage Examples

### Download Only

```bash
python training/download_datasets.py --all
```

### Optimize Hyperparameters Only

```bash
python training/hyperparameter_optimization.py --phishing
python training/hyperparameter_optimization.py --ai
```

### Train with Optimization

```bash
python training/train_phishing_model.py --optimize --dataset data/raw/phishing_emails.csv
python training/train_ai_detector.py --optimize --huggingface
```

### Complete Pipeline (All Steps)

```bash
python training/train_all_models.py --download --optimize --train
```

## Expected Outcomes

1. **Automated Downloads**: Both datasets downloaded automatically
2. **Optimized Models**: Best hyperparameters found automatically
3. **High Accuracy**: 

   - Phishing detection: >90% accuracy (with UTwente dataset)
   - AI detection: >95% accuracy (with Hugging Face dataset)

4. **Saved Models**: Ready to use in production
5. **Training Reports**: Metrics and evaluation results saved

## Implementation Order

1. Enhance download script (add UTwente download)
2. Create hyperparameter optimization module
3. Enhance training scripts with optimization
4. Create master training script
5. Add evaluation and reporting
6. Update documentation

## Notes

- Hyperparameter optimization may take 2-4 hours depending on dataset size
- Training