"""
Feature extraction utilities for different model types
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_nlp_features(text: str) -> Dict[str, Any]:
    """Extract features for NLP model"""
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'char_count': len(text),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
        'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0,
        'url_count': text.count('http://') + text.count('https://'),
        'email_count': text.count('@'),
    }
    return features


def extract_url_features(url_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features for URL/GNN model"""
    features = {
        'domain_length': len(url_data.get('domain', '')),
        'path_length': len(url_data.get('path', '')),
        'query_length': len(url_data.get('query', '')),
        'has_subdomain': '.' in url_data.get('domain', '').split('.')[0] if url_data.get('domain') else False,
        'port': url_data.get('port', 0),
        'is_https': url_data.get('scheme') == 'https',
    }
    return features


def extract_visual_features(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features for Visual/CNN model"""
    features = {
        'width': image_data.get('width', 0),
        'height': image_data.get('height', 0),
        'channels': image_data.get('channels', 3),
        'has_logo': image_data.get('has_logo', False),
        'form_count': image_data.get('form_count', 0),
        'link_count': image_data.get('link_count', 0),
    }
    return features


def extract_features_for_model_type(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """Extract features based on model type"""
    if model_type == 'nlp':
        if 'text' in df.columns:
            features_list = df['text'].apply(extract_nlp_features).tolist()
            features_df = pd.DataFrame(features_list)
            return pd.concat([df, features_df], axis=1)
    
    elif model_type == 'url':
        if 'input_data' in df.columns:
            features_list = df['input_data'].apply(
                lambda x: extract_url_features(x if isinstance(x, dict) else {})
            ).tolist()
            features_df = pd.DataFrame(features_list)
            return pd.concat([df, features_df], axis=1)
    
    elif model_type == 'visual':
        if 'input_data' in df.columns:
            features_list = df['input_data'].apply(
                lambda x: extract_visual_features(x if isinstance(x, dict) else {})
            ).tolist()
            features_df = pd.DataFrame(features_list)
            return pd.concat([df, features_df], axis=1)
    
    logger.warning(f"Unknown model type: {model_type}, returning original dataframe")
    return df
