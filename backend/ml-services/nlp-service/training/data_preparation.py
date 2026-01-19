"""
Data preparation utilities for training NLP models
"""
import os
import sys
import pandas as pd
import json
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import logging
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def prepare_phishing_dataset(csv_path: str, output_path: str):
    """
    Prepare phishing email dataset for training
    
    Expected CSV format:
    - text: Email content
    - label: 0 for legitimate, 1 for phishing
    """
    logger.info(f"Loading dataset from {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_columns = ['text', 'label']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    
    # Convert to Hugging Face dataset format
    dataset_dict = {
        'text': df['text'].tolist(),
        'label': df['label'].tolist()
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(dataset_dict, f, indent=2)
    
    logger.info(f"Dataset prepared and saved to {output_path}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Phishing samples: {df['label'].sum()}")
    logger.info(f"Legitimate samples: {len(df) - df['label'].sum()}")

def prepare_ai_detection_dataset(csv_path: str, output_path: str):
    """
    Prepare AI detection dataset for training
    
    Expected CSV format:
    - text: Text content
    - label: 0 for human-written, 1 for AI-generated
    """
    logger.info(f"Loading dataset from {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_columns = ['text', 'label']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    
    # Convert to Hugging Face dataset format
    dataset_dict = {
        'text': df['text'].tolist(),
        'label': df['label'].tolist()
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(dataset_dict, f, indent=2)
    
    logger.info(f"Dataset prepared and saved to {output_path}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"AI-generated samples: {df['label'].sum()}")
    logger.info(f"Human-written samples: {len(df) - df['label'].sum()}")

def preprocess_utwente_dataset(csv_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Preprocess UTwente phishing dataset
    
    Handles:
    - Safe/Phishing label mapping
    - Combining subject + body if separate columns
    - Text cleaning and normalization
    
    Args:
        csv_path: Path to UTwente dataset CSV
        output_path: Optional path to save processed dataset
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Preprocessing UTwente dataset from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Handle column mapping
    column_mapping = {
        'Email': 'text',
        'email': 'text',
        'body': 'text',
        'Body': 'text',
        'content': 'text',
        'Content': 'text',
        'message': 'text',
        'Message': 'text',
    }
    
    df = df.rename(columns=column_mapping)
    
    # Handle label mapping (Safe/Phishing -> 0/1)
    if 'label' in df.columns:
        if df['label'].dtype == 'object':
            label_mapping = {
                'Safe': 0,
                'safe': 0,
                'Legitimate': 0,
                'legitimate': 0,
                'Phishing': 1,
                'phishing': 1,
                'Malicious': 1,
                'malicious': 1
            }
            df['label'] = df['label'].map(label_mapping)
            df['label'] = df['label'].fillna(-1)
            df = df[df['label'] != -1]
    elif 'Label' in df.columns:
        label_mapping = {
            'Safe': 0,
            'safe': 0,
            'Legitimate': 0,
            'legitimate': 0,
            'Phishing': 1,
            'phishing': 1,
            'Malicious': 1,
            'malicious': 1
        }
        df['label'] = df['Label'].map(label_mapping)
        df['label'] = df['label'].fillna(-1)
        df = df[df['label'] != -1]
    
    # Combine subject and body if separate
    if 'subject' in df.columns and 'text' in df.columns:
        df['text'] = df['subject'].astype(str) + ' ' + df['text'].astype(str)
    elif 'Subject' in df.columns and 'text' in df.columns:
        df['text'] = df['Subject'].astype(str) + ' ' + df['text'].astype(str)
    
    # Text cleaning
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 10]
    
    # Ensure labels are 0 or 1
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    
    if output_path:
        df[['text', 'label']].to_csv(output_path, index=False)
        logger.info(f"Saved processed dataset to {output_path}")
    
    logger.info(f"Preprocessed {len(df)} samples")
    return df[['text', 'label']]

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (optional, can be customized)
    # text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()

def create_train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test split with stratification
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        stratify: Whether to stratify by label
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 0.01, "Sizes must sum to 1.0"
    
    logger.info(f"Creating train/val/test split: {train_size:.1%}/{val_size:.1%}/{test_size:.1%}")
    
    stratify_col = df['label'] if stratify else None
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_size),
        stratify=stratify_col,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    stratify_col_temp = temp_df['label'] if stratify else None
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=stratify_col_temp,
        random_state=random_state
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train labels - 0: {(train_df['label']==0).sum()}, 1: {(train_df['label']==1).sum()}")
    logger.info(f"Val labels - 0: {(val_df['label']==0).sum()}, 1: {(val_df['label']==1).sum()}")
    logger.info(f"Test labels - 0: {(test_df['label']==0).sum()}, 1: {(test_df['label']==1).sum()}")
    
    return train_df, val_df, test_df

def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Check data quality and return report
    
    Checks:
    - Class imbalance
    - Duplicate samples
    - Empty/malformed data
    - Text length distribution
    
    Returns:
        Dictionary with quality metrics
    """
    logger.info("Checking data quality...")
    
    report = {
        'total_samples': int(len(df)),
        'class_distribution': {},
        'duplicates': 0,
        'empty_texts': 0,
        'short_texts': 0,
        'text_length_stats': {}
    }
    
    # Class distribution
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().to_dict()
        # Convert numpy types to Python native types for JSON serialization
        report['class_distribution'] = {int(k): int(v) for k, v in label_counts.items()}
        
        # Check for imbalance
        if len(label_counts) == 2:
            ratio = min(label_counts.values()) / max(label_counts.values())
            report['class_imbalance_ratio'] = ratio
            if ratio < 0.3:
                logger.warning(f"Severe class imbalance detected: {ratio:.2f}")
    
    # Duplicates
    if 'text' in df.columns:
        duplicates = int(df['text'].duplicated().sum())
        report['duplicates'] = duplicates
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate texts")
    
    # Empty texts
    if 'text' in df.columns:
        empty = int(df['text'].isna().sum() + (df['text'].str.len() == 0).sum())
        report['empty_texts'] = empty
        
        # Short texts (< 10 chars)
        short = int((df['text'].str.len() < 10).sum())
        report['short_texts'] = short
        
        # Text length statistics
        text_lengths = df['text'].str.len()
        report['text_length_stats'] = {
            'mean': float(text_lengths.mean()),
            'median': float(text_lengths.median()),
            'min': int(text_lengths.min()),
            'max': int(text_lengths.max()),
            'std': float(text_lengths.std())
        }
    
    logger.info(f"Quality check complete: {report['total_samples']} samples")
    return report

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate samples"""
    initial_len = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    removed = initial_len - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate samples")
    return df

if __name__ == "__main__":
    # Example usage
    # prepare_phishing_dataset("data/phishing_emails.csv", "data/phishing_dataset.json")
    # prepare_ai_detection_dataset("data/ai_texts.csv", "data/ai_dataset.json")
    logger.info("Data preparation utilities ready. Import and use as needed.")
