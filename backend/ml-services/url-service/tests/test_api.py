"""
Tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "url-service"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "url-service"

def test_analyze_url():
    response = client.post(
        "/api/v1/analyze-url",
        json={"url": "https://example.com"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "url_analysis" in data
    assert "domain_analysis" in data
    assert "processing_time_ms" in data

def test_analyze_domain():
    response = client.post(
        "/api/v1/analyze-domain",
        json={"domain": "example.com"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "example.com"
    assert "analysis" in data

def test_invalid_url():
    response = client.post(
        "/api/v1/analyze-url",
        json={"url": "not-a-valid-url"}
    )
    # Should either return 400 or sanitize the URL
    assert response.status_code in [200, 400]
