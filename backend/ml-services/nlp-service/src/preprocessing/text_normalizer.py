import re
import html
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

class TextNormalizer:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not available, using empty set")
            self.stop_words = set()
    
    def normalize(self, text: str) -> str:
        """Normalize text for analysis"""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (but keep placeholder)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        return text.strip()
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        try:
            words = word_tokenize(text.lower())
            filtered_words = [word for word in words if word not in self.stop_words]
            return ' '.join(filtered_words)
        except Exception as e:
            logger.warning(f"Failed to remove stopwords: {e}")
            return text
    
    def clean_text(self, text: str, remove_stopwords: bool = False) -> str:
        """Clean and normalize text"""
        normalized = self.normalize(text)
        if remove_stopwords:
            normalized = self.remove_stopwords(normalized)
        return normalized
