import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, MagicMock
from src.main import app
from src.utils.model_loader import ModelLoader

# Create test client
client = TestClient(app)

# Setup mock model loader for all tests
@pytest.fixture(autouse=True)
def setup_mocks():
    """Setup mocks for all tests"""
    mock_loader = Mock(spec=ModelLoader)
    
    # Mock phishing classifier
    mock_phishing = Mock()
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
    mock_ai = Mock()
    mock_ai.detect.return_value = {
        "ai_generated_probability": 0.30,
        "human_written_probability": 0.70,
        "is_ai_generated": False,
        "confidence": 0.40
    }
    
    mock_loader.phishing_classifier = mock_phishing
    mock_loader.ai_detector = mock_ai
    mock_loader.is_loaded.return_value = True
    
    # Patch ModelLoader.get_instance and get_models dependency
    with patch.object(ModelLoader, 'get_instance', return_value=mock_loader):
        with patch('src.api.routes.get_models', return_value=mock_loader):
            with patch('src.main.model_loader', mock_loader):
                yield

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "nlp-service"

def test_analyze_text():
    """Test text analysis endpoint"""
    response = client.post("/api/v1/analyze-text", json={
        "text": "Urgent! Verify your account immediately or it will be suspended.",
        "include_features": False
    })
    assert response.status_code == 200
    data = response.json()
    assert "phishing_probability" in data
    assert "prediction" in data
    assert "processing_time_ms" in data
    # Should detect as phishing (from mock)
    assert data["phishing_probability"] > 0.3

def test_analyze_text_with_features():
    """Test text analysis with features"""
    response = client.post("/api/v1/analyze-text", json={
        "text": "This is a normal email message.",
        "include_features": True
    })
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert data["features"] is not None

def test_analyze_email():
    """Test email analysis endpoint"""
    sample_email = """From: test@example.com
To: user@example.com
Subject: Urgent Account Verification Required

Dear User,

Your account has been suspended. Please verify immediately by clicking the link below.

Thank you.
"""
    response = client.post("/api/v1/analyze-email", json={
        "raw_email": sample_email,
        "include_features": False
    })
    assert response.status_code == 200
    data = response.json()
    assert "phishing_probability" in data
    assert "prediction" in data

def test_detect_ai_content():
    """Test AI content detection endpoint"""
    response = client.post("/api/v1/detect-ai-content?text=This is a test message.")
    assert response.status_code == 200
    data = response.json()
    assert "ai_generated_probability" in data
    assert "is_ai_generated" in data
    assert "confidence" in data
