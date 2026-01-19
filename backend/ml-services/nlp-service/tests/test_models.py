import pytest
from src.models.phishing_classifier import PhishingClassifier
from src.models.ai_detector import AIGeneratedDetector
from src.models.feature_extractor import FeatureExtractor

def test_phishing_classifier():
    """Test phishing classifier"""
    classifier = PhishingClassifier()
    
    # Test phishing text
    result = classifier.predict("Urgent! Verify your account immediately or it will be suspended.")
    assert "phishing_probability" in result
    assert "prediction" in result
    assert result["phishing_probability"] >= 0.0
    assert result["phishing_probability"] <= 1.0
    
    # Test legitimate text
    result2 = classifier.predict("This is a normal email about a meeting scheduled for next week.")
    assert result2["phishing_probability"] >= 0.0
    assert result2["phishing_probability"] <= 1.0

def test_ai_detector():
    """Test AI detector"""
    detector = AIGeneratedDetector()
    
    result = detector.detect("This is a test message to check if it's AI generated.")
    assert "ai_generated_probability" in result
    assert "is_ai_generated" in result
    assert "confidence" in result

def test_feature_extractor():
    """Test feature extractor"""
    extractor = FeatureExtractor()
    
    text = "Urgent! Click here to verify your account: http://example.com"
    features = extractor.extract_linguistic_features(text)
    
    assert "keyword_matches" in features
    assert "has_urls" in features
    assert features["has_urls"] == True
    
    urls = extractor.extract_urls(text)
    assert len(urls) > 0
    
    emails = extractor.extract_emails("Contact us at support@example.com")
    assert len(emails) > 0
