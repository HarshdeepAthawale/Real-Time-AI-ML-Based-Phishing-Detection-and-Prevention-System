"""Sentiment analysis for detecting fear/urgency tactics"""
from typing import Dict
from src.utils.logger import logger


class SentimentAnalyzer:
    """Analyze sentiment to detect fear-based tactics"""
    
    def __init__(self):
        # Fear/threat keywords
        self.negative_keywords = [
            'suspend', 'locked', 'blocked', 'unauthorized', 'fraud', 'illegal',
            'criminal', 'police', 'jail', 'arrest', 'penalty', 'fine', 'lawsuit',
            'compromise', 'breach', 'hacked', 'stolen', 'theft'
        ]
        
        # Positive/reassuring keywords
        self.positive_keywords = [
            'secure', 'safe', 'protect', 'verify', 'confirm', 'update'
        ]
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis
        """
        text_lower = text.lower()
        
        # Count negative/fear keywords
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        # Count positive keywords (often used deceptively)
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        total_keywords = negative_count + positive_count
        if total_keywords > 0:
            sentiment_score = (positive_count - negative_count) / total_keywords
        else:
            sentiment_score = 0.0
        
        # Determine overall sentiment
        if sentiment_score < -0.3:
            sentiment_label = "NEGATIVE"
        elif sentiment_score > 0.3:
            sentiment_label = "POSITIVE"
        else:
            sentiment_label = "NEUTRAL"
        
        # High negative count is suspicious
        is_negative = negative_count > 3
        
        # Mix of fear + reassurance is very suspicious
        is_mixed_tactics = negative_count >= 2 and positive_count >= 2
        
        return {
            "sentiment": sentiment_label,
            "sentiment_score": sentiment_score,
            "negative_keyword_count": negative_count,
            "positive_keyword_count": positive_count,
            "is_negative": is_negative,
            "uses_fear_tactics": negative_count > 2,
            "uses_mixed_tactics": is_mixed_tactics,
            "suspicion_level": "high" if is_mixed_tactics else ("medium" if is_negative else "low")
        }
