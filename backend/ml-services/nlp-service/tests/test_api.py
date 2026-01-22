"""API endpoint tests"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "nlp-service"


def test_analyze_text():
    """Test text analysis endpoint"""
    response = client.post("/api/v1/analyze-text", json={
        "text": "Urgent! Your account will be suspended. Click here to verify.",
        "include_features": False
    })
    assert response.status_code == 200
    data = response.json()
    assert "phishing_probability" in data
    assert "urgency_score" in data
    assert "sentiment" in data


def test_analyze_email():
    """Test email analysis endpoint"""
    response = client.post("/api/v1/analyze-email", json={
        "subject": "Verify your account immediately",
        "body": "Your account will be suspended unless you verify now.",
        "sender": "noreply@suspicious.com"
    })
    assert response.status_code == 200
    data = response.json()
    assert "phishing_probability" in data


def test_detect_ai_content():
    """Test AI content detection endpoint"""
    response = client.post("/api/v1/detect-ai-content", json={
        "text": "This is a test message to check AI detection."
    })
    assert response.status_code == 200
    data = response.json()
    assert "ai_generated_probability" in data


def test_model_info():
    """Test model info endpoint"""
    response = client.get("/api/v1/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "phishing_classifier" in data
    assert "ai_detector" in data
