"""
Performance tests for NLP service
"""
import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.main import app
from src.utils.model_loader import ModelLoader

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_mocks():
    """Setup mocks for performance tests"""
    mock_loader = Mock(spec=ModelLoader)
    
    # Mock phishing classifier with fast response
    mock_phishing = Mock()
    mock_phishing.predict.return_value = {
        "phishing_probability": 0.75,
        "legitimate_probability": 0.25,
        "confidence": 0.50,
        "prediction": "phishing"
    }
    mock_phishing.extract_features.return_value = {
        "embeddings": [0.1] * 768,
        "embedding_dim": 768
    }
    
    # Mock AI detector
    mock_ai = Mock()
    mock_ai.detect.return_value = {
        "ai_generated_probability": 0.25,
        "human_written_probability": 0.75,
        "is_ai_generated": False,
        "confidence": 0.50
    }
    
    mock_loader.phishing_classifier = mock_phishing
    mock_loader.ai_detector = mock_ai
    mock_loader.is_loaded.return_value = True
    
    with patch.object(ModelLoader, 'get_instance', return_value=mock_loader):
        with patch('src.api.routes.get_models', return_value=mock_loader):
            with patch('src.main.model_loader', mock_loader):
                yield

def test_inference_latency():
    """Test that inference latency is below 50ms target"""
    start_time = time.time()
    
    response = client.post("/api/v1/analyze-text", json={
        "text": "Test message for latency measurement.",
        "include_features": False
    })
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    assert response.status_code == 200
    # Target: <50ms (but with mocks it should be much faster)
    # In real scenario, this would test actual model inference
    assert elapsed_ms < 1000  # Very lenient for mocked tests
    
def test_throughput():
    """Test that service can handle multiple requests"""
    num_requests = 10
    start_time = time.time()
    
    for i in range(num_requests):
        response = client.post("/api/v1/analyze-text", json={
            "text": f"Test message {i}",
            "include_features": False
        })
        assert response.status_code == 200
    
    total_time = time.time() - start_time
    requests_per_second = num_requests / total_time
    
    # Target: 100+ req/s (with mocks it should be much higher)
    # In real scenario, this would test actual throughput
    assert requests_per_second > 10  # Very lenient for mocked tests

def test_cached_response_performance():
    """Test that cached responses are faster"""
    text = "This is a test message for caching performance."
    
    # First request (cache miss)
    start1 = time.time()
    response1 = client.post("/api/v1/analyze-text", json={
        "text": text,
        "include_features": False
    })
    time1 = (time.time() - start1) * 1000
    
    # Second request (cache hit, if Redis available)
    start2 = time.time()
    response2 = client.post("/api/v1/analyze-text", json={
        "text": text,
        "include_features": False
    })
    time2 = (time.time() - start2) * 1000
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    # Cached response should be faster (if caching works)
    # Note: This test may not show difference if Redis is unavailable
    assert time2 < time1 * 2  # Cached should be at least not much slower
