"""Feature extraction utilities"""
from typing import Dict, List


class FeatureExtractor:
    """Extract statistical features from text"""
    
    def extract_text_features(self, text: str) -> Dict:
        """
        Extract statistical features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text features
        """
        words = text.split()
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            "special_char_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0,
            "exclamation_count": text.count('!'),
            "question_count": text.count('?'),
            "url_count": text.count('[URL]'),
            "email_count": text.count('[EMAIL]'),
            "phone_count": text.count('[PHONE]')
        }
    
    def extract_email_features(self, email_data: Dict) -> Dict:
        """Extract features from parsed email"""
        features = {}
        
        # Subject features
        subject = email_data.get('subject', '')
        features['subject_length'] = len(subject)
        features['subject_has_urgency'] = any(
            word in subject.lower() 
            for word in ['urgent', 'immediate', 'action required', 'verify', 'suspended']
        )
        
        # Sender features
        from_addr = email_data.get('from', '')
        reply_to = email_data.get('reply_to', '')
        features['has_reply_to'] = bool(reply_to)
        features['reply_to_mismatch'] = reply_to != from_addr if reply_to else False
        
        # Body features
        body = email_data.get('body_text', '')
        features.update(self.extract_text_features(body))
        
        # Attachment feature
        features['has_attachments'] = email_data.get('has_attachments', False)
        
        return features
