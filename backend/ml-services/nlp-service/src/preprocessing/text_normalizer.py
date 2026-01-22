"""Text normalization utilities"""
import re
import html
from typing import List


class TextNormalizer:
    """Normalize text for ML analysis"""
    
    def __init__(self):
        # URL pattern
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        # Email pattern
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        # Phone pattern
        self.phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    
    def normalize(self, text: str) -> str:
        """
        Normalize text for analysis
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text
        """
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Replace URLs with placeholder
        text = re.sub(self.url_pattern, '[URL]', text)
        
        # Replace emails with placeholder
        text = re.sub(self.email_pattern, '[EMAIL]', text)
        
        # Replace phone numbers with placeholder
        text = re.sub(self.phone_pattern, '[PHONE]', text)
        
        return text.strip()
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        return re.findall(self.url_pattern, text)
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        return re.findall(self.email_pattern, text)
    
    def clean_for_display(self, text: str) -> str:
        """Clean text for display (less aggressive than normalize)"""
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
