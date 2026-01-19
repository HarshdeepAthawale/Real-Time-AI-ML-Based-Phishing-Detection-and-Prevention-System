from typing import List, Dict
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

class CustomTokenizer:
    """Custom tokenization utilities"""
    
    def __init__(self):
        pass
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            return word_tokenize(text)
        except Exception as e:
            logger.warning(f"Failed to tokenize words: {e}")
            # Fallback to simple whitespace tokenization
            return text.split()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        try:
            return sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Failed to tokenize sentences: {e}")
            # Fallback to simple sentence splitting
            return re.split(r'[.!?]+', text)
    
    def tokenize_with_positions(self, text: str) -> List[Dict]:
        """Tokenize text and return tokens with their positions"""
        tokens = self.tokenize_words(text)
        positions = []
        current_pos = 0
        
        for token in tokens:
            pos = text.find(token, current_pos)
            if pos != -1:
                positions.append({
                    "token": token,
                    "start": pos,
                    "end": pos + len(token)
                })
                current_pos = pos + len(token)
            else:
                positions.append({
                    "token": token,
                    "start": current_pos,
                    "end": current_pos + len(token)
                })
                current_pos += len(token)
        
        return positions
    
    def ngram_tokenize(self, text: str, n: int = 2) -> List[str]:
        """Generate n-grams from text"""
        tokens = self.tokenize_words(text)
        ngrams = []
        
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def clean_tokens(self, tokens: List[str]) -> List[str]:
        """Clean tokens by removing punctuation and lowercasing"""
        cleaned = []
        for token in tokens:
            # Remove punctuation
            cleaned_token = re.sub(r'[^\w\s]', '', token)
            if cleaned_token:
                cleaned.append(cleaned_token.lower())
        return cleaned
