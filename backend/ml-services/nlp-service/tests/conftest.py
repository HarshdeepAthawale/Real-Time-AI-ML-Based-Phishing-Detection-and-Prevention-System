"""
Pytest configuration and fixtures for NLP service tests
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.utils.model_loader import ModelLoader
from src.models.phishing_classifier import PhishingClassifier
from src.models.ai_detector import AIGeneratedDetector

@pytest.fixture
def mock_model_loader():
    """Create a mock ModelLoader for testing"""
    mock_loader = Mock(spec=ModelLoader)
    
    # Mock phishing classifier
    mock_phishing = Mock(spec=PhishingClassifier)
    mock_phishing.predict.return_value = {
        "phishing_probability": 0.85,
        "legitimate_probability": 0.15,
        "confidence": 0.70,
        "prediction": "phishing"
    }
    mock_phishing.extract_features.return_value = {
        "embeddings": [0.1] * 768,
        "embedding_dim": 768
    }
    
    # Mock AI detector
    mock_ai = Mock(spec=AIGeneratedDetector)
    mock_ai.detect.return_value = {
        "ai_generated_probability": 0.30,
        "human_written_probability": 0.70,
        "is_ai_generated": False,
        "confidence": 0.40
    }
    
    mock_loader.phishing_classifier = mock_phishing
    mock_loader.ai_detector = mock_ai
    mock_loader.is_loaded.return_value = True
    
    return mock_loader

@pytest.fixture
def mock_phishing_classifier():
    """Create a mock PhishingClassifier"""
    mock = Mock(spec=PhishingClassifier)
    mock.predict.return_value = {
        "phishing_probability": 0.75,
        "legitimate_probability": 0.25,
        "confidence": 0.50,
        "prediction": "phishing"
    }
    mock.extract_features.return_value = {
        "embeddings": [0.1] * 768,
        "embedding_dim": 768
    }
    return mock

@pytest.fixture
def mock_ai_detector():
    """Create a mock AIGeneratedDetector"""
    mock = Mock(spec=AIGeneratedDetector)
    mock.detect.return_value = {
        "ai_generated_probability": 0.25,
        "human_written_probability": 0.75,
        "is_ai_generated": False,
        "confidence": 0.50
    }
    return mock

@pytest.fixture(autouse=True)
def mock_model_loader_instance(mock_model_loader):
    """Automatically patch ModelLoader.get_instance() for all tests"""
    # Patch both the get_instance method and the get_models dependency
    with patch('src.utils.model_loader.ModelLoader.get_instance', return_value=mock_model_loader):
        with patch('src.api.routes.get_models') as mock_get_models:
            mock_get_models.return_value = mock_model_loader
            # Also patch the main.py model_loader
            with patch('src.main.model_loader', mock_model_loader):
                yield mock_model_loader
