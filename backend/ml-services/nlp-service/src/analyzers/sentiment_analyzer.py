from transformers import pipeline
from typing import Dict
import logging
import os

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = None
        try:
            # Use a pre-trained sentiment analysis model
            # This model is lightweight and works well for general sentiment
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=-1  # Use CPU (-1) or GPU (0, 1, etc.)
            )
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}. Using fallback method.")
            self.sentiment_pipeline = None
    
    def analyze(self, text: str) -> Dict:
        """Analyze sentiment"""
        if not text:
            return {
                "sentiment": "NEUTRAL",
                "sentiment_score": 0.5,
                "is_negative": False
            }
        
        if self.sentiment_pipeline is None:
            return self._fallback_sentiment(text)
        
        try:
            # Truncate for model limit (most models have 512 token limit)
            truncated_text = text[:512] if len(text) > 512 else text
            result = self.sentiment_pipeline(truncated_text)[0]
            
            # Map model output to our format
            label = result['label'].upper()
            score = result['score']
            
            # Handle different label formats
            is_negative = label in ['NEGATIVE', 'LABEL_0', '0']
            
            return {
                "sentiment": label,
                "sentiment_score": score,
                "is_negative": is_negative
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> Dict:
        """Fallback sentiment analysis using keyword matching"""
        text_lower = text.lower()
        
        # Negative sentiment keywords
        negative_keywords = [
            'urgent', 'suspended', 'locked', 'expired', 'threat', 'warning',
            'alert', 'danger', 'critical', 'immediate', 'action required',
            'unauthorized', 'breach', 'compromised'
        ]
        
        # Positive sentiment keywords
        positive_keywords = [
            'congratulations', 'winner', 'prize', 'reward', 'bonus',
            'free', 'special offer', 'exclusive'
        ]
        
        negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
        positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
        
        if negative_count > positive_count:
            sentiment = "NEGATIVE"
            score = min(0.9, 0.5 + (negative_count * 0.1))
        elif positive_count > negative_count:
            sentiment = "POSITIVE"
            score = min(0.9, 0.5 + (positive_count * 0.1))
        else:
            sentiment = "NEUTRAL"
            score = 0.5
        
        return {
            "sentiment": sentiment,
            "sentiment_score": score,
            "is_negative": sentiment == "NEGATIVE"
        }
