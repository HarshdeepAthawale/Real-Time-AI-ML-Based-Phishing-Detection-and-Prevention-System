"""Urgency detection analyzer"""
import re
from typing import Dict, List


class UrgencyAnalyzer:
    """Detect urgency indicators in text"""
    
    def __init__(self):
        # Urgency keywords
        self.urgency_keywords = [
            'urgent', 'immediately', 'asap', 'hurry', 'expired', 'expiring',
            'suspended', 'locked', 'verify', 'confirm', 'update', 'action required',
            'limited time', 'act now', 'click here', 'verify account', 'unusual activity',
            'security alert', 'account suspended', 'verify identity', 'confirm identity'
        ]
        
        # Time-based urgency patterns
        self.time_patterns = [
            r'\d+\s*(hour|day|minute)s?\s*(ago|left|remaining)',
            r'expir(es|ed|ing)',
            r'deadline',
            r'within\s+\d+\s*(hour|day|minute)s?',
            r'before\s+\d+',
            r'until\s+\d+'
        ]
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze urgency indicators in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with urgency analysis results
        """
        text_lower = text.lower()
        
        # Count urgency keywords
        keyword_matches = []
        keyword_count = 0
        for keyword in self.urgency_keywords:
            if keyword in text_lower:
                keyword_count += 1
                keyword_matches.append(keyword)
        
        # Check time patterns
        time_pattern_matches = []
        for pattern in self.time_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                time_pattern_matches.extend(matches)
        
        # Calculate urgency score (0-100)
        urgency_score = min(100, (keyword_count * 15) + (len(time_pattern_matches) * 20))
        
        # Check for excessive punctuation (urgency indicator)
        exclamation_count = text.count('!')
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Increase score for excessive punctuation
        if exclamation_count > 3:
            urgency_score = min(100, urgency_score + 10)
        if caps_words > 5:
            urgency_score = min(100, urgency_score + 10)
        
        return {
            "urgency_score": urgency_score,
            "keyword_count": keyword_count,
            "keyword_matches": keyword_matches[:5],  # Top 5
            "time_pattern_count": len(time_pattern_matches),
            "exclamation_count": exclamation_count,
            "caps_words_count": caps_words,
            "is_urgent": urgency_score > 50
        }
