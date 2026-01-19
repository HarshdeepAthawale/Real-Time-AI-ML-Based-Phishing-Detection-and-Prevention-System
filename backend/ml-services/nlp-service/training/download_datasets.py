"""
Helper script to download datasets for training
"""
import os
import pandas as pd
import requests
from datasets import load_dataset
import logging
from typing import Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_ai_detection_dataset(max_retries: int = 3) -> Optional[str]:
    """Download AI detection dataset from Hugging Face with progress tracking"""
    logger.info("Downloading AI detection dataset from Hugging Face...")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}...")
            dataset = load_dataset("shahxeebhassan/human_vs_ai_sentences")
            
            # Create data directory
            os.makedirs("data/raw", exist_ok=True)
            
            # Convert to DataFrame and save
            logger.info("Converting dataset to DataFrame...")
            df = pd.DataFrame(dataset['train'])
            
            # Ensure correct column names
            if 'sentence' in df.columns and 'text' not in df.columns:
                df['text'] = df['sentence']
            if 'generated' in df.columns and 'label' not in df.columns:
                df['label'] = df['generated'].astype(int)
            
            # Validate data
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset missing required columns: 'text' and 'label'")
            
            # Save as CSV
            output_path = "data/raw/ai_detection.csv"
            df[['text', 'label']].to_csv(output_path, index=False)
            
            logger.info(f"✓ Downloaded {len(df)} samples")
            logger.info(f"✓ Saved to {output_path}")
            logger.info(f"✓ AI-generated: {df['label'].sum()}, Human-written: {len(df) - df['label'].sum()}")
            
            # Validate download
            if validate_dataset(output_path):
                return output_path
            else:
                raise ValueError("Dataset validation failed")
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download dataset after {max_retries} attempts")
                logger.info("Make sure you have 'datasets' library installed: pip install datasets")
                return None
    
    return None

def download_utwente_phishing_dataset(max_retries: int = 3) -> Optional[str]:
    """Download UTwente phishing dataset from Zenodo API"""
    logger.info("Downloading UTwente phishing dataset from Zenodo...")
    
    zenodo_record_id = "13474746"
    api_url = f"https://zenodo.org/api/records/{zenodo_record_id}/files"
    output_path = "data/raw/phishing_emails.csv"
    
    # Create data directory
    os.makedirs("data/raw", exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}...")
            logger.info(f"Fetching file list from Zenodo API...")
            
            # Get file list from Zenodo API
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Zenodo API returns data in different formats - handle both
            if isinstance(data, dict):
                files = data.get('files', [])
            elif isinstance(data, list):
                files = data
            else:
                raise ValueError(f"Unexpected API response format: {type(data)}")
            
            # Find the CSV file
            csv_file = None
            for file_info in files:
                # Handle both dict and different key names
                if isinstance(file_info, dict):
                    file_key = file_info.get('key') or file_info.get('filename') or file_info.get('name', '')
                else:
                    continue
                    
                if 'Phishing_validation_emails.csv' in file_key or 'phishing' in file_key.lower():
                    csv_file = file_info
                    break
            
            # Determine download URL
            download_url = None
            if csv_file:
                # Try to get download URL from API response
                if isinstance(csv_file, dict):
                    download_url = csv_file.get('links', {}).get('download') or csv_file.get('download_url') or csv_file.get('url')
            
            # If no download URL from API, use direct URL
            if not download_url:
                download_url = f"https://zenodo.org/record/{zenodo_record_id}/files/Phishing_validation_emails.csv"
                logger.info(f"Using direct download URL: {download_url}")
            
            # Download the CSV file
            logger.info(f"Downloading from: {download_url}")
            csv_response = requests.get(download_url, timeout=60, stream=True, allow_redirects=True)
            csv_response.raise_for_status()
            
            # Save file
            total_size = int(csv_response.headers.get('content-length', 0))
            if total_size > 0:
                logger.info(f"File size: {total_size / 1024:.2f} KB")
            
            with open(output_path, 'wb') as f:
                for chunk in csv_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✓ Downloaded CSV file")
            
            # Load and preprocess the dataset
            logger.info("Preprocessing dataset...")
            df = pd.read_csv(output_path)
            
            # Handle column mapping
            # UTwente dataset has specific column names: 'Email Text' and 'Email Type'
            # Map to standard format
            column_mapping = {
                'Email Text': 'text',
                'Email': 'text',
                'email': 'text',
                'body': 'text',
                'Body': 'text',
                'content': 'text',
                'Content': 'text',
                'Email Type': 'label',
                'Label': 'label',
                'label': 'label',
                'Type': 'label',
                'type': 'label',
                'Class': 'label',
                'class': 'label',
            }
            
            # Rename columns if needed
            df = df.rename(columns=column_mapping)
            
            # Handle label mapping (Safe Email/Phishing Email -> 0/1)
            if 'label' in df.columns:
                if df['label'].dtype == 'object':
                    label_mapping = {
                        'Safe Email': 0,
                        'Safe': 0,
                        'safe': 0,
                        'safe email': 0,
                        'Legitimate': 0,
                        'legitimate': 0,
                        'Phishing Email': 1,
                        'Phishing': 1,
                        'phishing': 1,
                        'phishing email': 1,
                        'Malicious': 1,
                        'malicious': 1
                    }
                    df['label'] = df['label'].map(label_mapping)
                    # Fill any unmapped values
                    df['label'] = df['label'].fillna(-1)
                    df = df[df['label'] != -1]
            
            # Combine subject and body if separate
            if 'subject' in df.columns and 'text' in df.columns:
                df['text'] = df['subject'].astype(str) + ' ' + df['text'].astype(str)
            elif 'Subject' in df.columns and 'text' in df.columns:
                df['text'] = df['Subject'].astype(str) + ' ' + df['text'].astype(str)
            
            # Ensure we have text and label columns
            if 'text' not in df.columns:
                # Try to find text column
                possible_text_cols = [col for col in df.columns if col.lower() in ['email', 'body', 'content', 'message']]
                if possible_text_cols:
                    df['text'] = df[possible_text_cols[0]]
                else:
                    raise ValueError("Could not find 'text' column in dataset")
            
            if 'label' not in df.columns:
                raise ValueError("Could not find 'label' column in dataset")
            
            # Clean data
            df = df.dropna(subset=['text', 'label'])
            df = df[df['text'].str.len() > 10]  # Remove very short texts
            
            # Ensure labels are 0 or 1
            df['label'] = df['label'].astype(int)
            df = df[df['label'].isin([0, 1])]
            
            # Save processed dataset
            df[['text', 'label']].to_csv(output_path, index=False)
            
            logger.info(f"✓ Processed {len(df)} samples")
            logger.info(f"✓ Saved to {output_path}")
            logger.info(f"✓ Phishing: {df['label'].sum()}, Legitimate: {len(df) - df['label'].sum()}")
            
            # Validate download
            if validate_dataset(output_path):
                return output_path
            else:
                raise ValueError("Dataset validation failed")
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download dataset after {max_retries} attempts")
                return None
    
    return None

def validate_dataset(csv_path: str) -> bool:
    """Validate downloaded dataset"""
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            logger.error("Dataset missing required columns: 'text' and 'label'")
            return False
        
        # Check data quality
        if len(df) == 0:
            logger.error("Dataset is empty")
            return False
        
        # Check for class balance
        label_counts = df['label'].value_counts()
        if len(label_counts) < 2:
            logger.warning("Dataset has only one class")
        
        # Check for empty texts
        empty_texts = df['text'].isna().sum() + (df['text'].str.len() == 0).sum()
        if empty_texts > 0:
            logger.warning(f"Found {empty_texts} empty texts")
        
        logger.info(f"✓ Dataset validation passed: {len(df)} samples")
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False

def download_all_datasets() -> dict:
    """Download all datasets"""
    logger.info("=" * 60)
    logger.info("DOWNLOADING ALL DATASETS")
    logger.info("=" * 60)
    
    results = {
        'ai_detection': None,
        'phishing': None
    }
    
    # Download AI detection dataset
    logger.info("\n[1/2] Downloading AI detection dataset...")
    results['ai_detection'] = download_ai_detection_dataset()
    
    # Download phishing dataset
    logger.info("\n[2/2] Downloading phishing dataset...")
    results['phishing'] = download_utwente_phishing_dataset()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"AI Detection Dataset: {'✓ Downloaded' if results['ai_detection'] else '✗ Failed'}")
    logger.info(f"Phishing Dataset: {'✓ Downloaded' if results['phishing'] else '✗ Failed'}")
    
    return results

def download_phishing_dataset_info():
    """Print information about where to download phishing datasets"""
    logger.info("=" * 60)
    logger.info("PHISHING EMAIL DATASET SOURCES")
    logger.info("=" * 60)
    logger.info("")
    logger.info("1. Sting9 Research Initiative (Recommended)")
    logger.info("   URL: https://sting9.org/dataset")
    logger.info("   License: CC0 (Public Domain)")
    logger.info("   Format: SQL dump, REST API, or CSV")
    logger.info("")
    logger.info("2. Seven Phishing Email Datasets (Figshare)")
    logger.info("   URL: https://figshare.com/articles/dataset/Curated_Dataset_-_Phishing_Email/24899952")
    logger.info("   Size: ~203,000 emails")
    logger.info("   Format: CSV")
    logger.info("")
    logger.info("3. UTwente Phishing Validation Dataset")
    logger.info("   URL: https://research.utwente.nl/en/datasets/phishing-validation-emails-dataset/")
    logger.info("   Size: 2,000 emails")
    logger.info("   Format: CSV")
    logger.info("")
    logger.info("After downloading, save as: data/raw/phishing_emails.csv")
    logger.info("Required columns: 'text' (email content) and 'label' (0=legitimate, 1=phishing)")
    logger.info("")
    logger.info("See DATASET_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for training")
    parser.add_argument("--ai", action="store_true", help="Download AI detection dataset")
    parser.add_argument("--phishing", action="store_true", help="Download UTwente phishing dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--phishing-info", action="store_true", help="Show phishing dataset sources")
    
    args = parser.parse_args()
    
    if args.all:
        download_all_datasets()
    elif args.ai:
        download_ai_detection_dataset()
    elif args.phishing:
        download_utwente_phishing_dataset()
    elif args.phishing_info:
        download_phishing_dataset_info()
    else:
        logger.info("Usage:")
        logger.info("  python download_datasets.py --all              # Download all datasets")
        logger.info("  python download_datasets.py --ai              # Download AI detection dataset")
        logger.info("  python download_datasets.py --phishing         # Download phishing dataset")
        logger.info("  python download_datasets.py --phishing-info    # Show phishing dataset sources")
