"""
Tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import base64
from PIL import Image
import io

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "visual-service"
    assert "models_loaded" in data

def test_root(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "visual-service"
    assert data["version"] == "1.0.0"

def test_analyze_page(client, mock_page_renderer, sample_html):
    """Test analyze-page endpoint"""
    # Update mock to return our sample HTML
    mock_page_renderer.render_page.return_value = {
        "screenshot": base64.b64encode(b"fake_screenshot").decode(),
        "dom": sample_html,
        "status_code": 200,
        "metrics": {
            "width": 1920,
            "height": 1080,
            "scrollHeight": 1080,
            "elementCount": 10
        },
        "url": "https://example.com"
    }
    
    response = client.post(
        "/api/v1/analyze-page",
        json={
            "url": "https://example.com",
            "wait_time": 3000
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["url"] == "https://example.com"
    assert "dom_analysis" in data
    assert "form_analysis" in data
    assert "css_analysis" in data
    assert "processing_time_ms" in data
    assert isinstance(data["processing_time_ms"], (int, float))
    
    # Verify DOM analysis structure
    dom_analysis = data["dom_analysis"]
    assert "dom_hash" in dom_analysis
    assert "element_count" in dom_analysis
    assert "forms" in dom_analysis
    assert "links" in dom_analysis
    assert "images" in dom_analysis
    assert "structure" in dom_analysis
    assert "is_suspicious" in dom_analysis
    
    # Verify form analysis structure
    form_analysis = data["form_analysis"]
    assert "form_count" in form_analysis
    assert "credential_harvesting_score" in form_analysis
    assert "suspicious_patterns" in form_analysis
    assert "is_suspicious" in form_analysis

def test_analyze_page_with_legitimate_url(client, mock_page_renderer, sample_html):
    """Test analyze-page endpoint with legitimate URL for comparison"""
    mock_page_renderer.render_page.return_value = {
        "screenshot": base64.b64encode(b"fake_screenshot").decode(),
        "dom": sample_html,
        "status_code": 200,
        "metrics": {"width": 1920, "height": 1080, "scrollHeight": 1080, "elementCount": 10},
        "url": "https://example.com"
    }
    
    response = client.post(
        "/api/v1/analyze-page",
        json={
            "url": "https://example.com",
            "legitimate_url": "https://legitimate.com"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "similarity_analysis" in data
    # Similarity analysis should be present when legitimate_url is provided
    if data["similarity_analysis"]:
        assert "similarity_score" in data["similarity_analysis"]
        assert "hamming_distance" in data["similarity_analysis"]
        assert "is_similar" in data["similarity_analysis"]

def test_analyze_page_rendering_error(client, mock_page_renderer):
    """Test analyze-page endpoint with rendering error"""
    mock_page_renderer.render_page.return_value = {
        "error": "Failed to load page",
        "url": "https://invalid-url.com"
    }
    
    response = client.post(
        "/api/v1/analyze-page",
        json={"url": "https://invalid-url.com"}
    )
    
    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower() or "failed" in response.json()["detail"].lower()

def test_compare_visual(client, mock_page_renderer):
    """Test compare-visual endpoint"""
    # Create two different mock screenshots
    img1 = Image.new('RGB', (100, 100), color='red')
    img1_bytes = io.BytesIO()
    img1.save(img1_bytes, format='PNG')
    
    img2 = Image.new('RGB', (100, 100), color='blue')
    img2_bytes = io.BytesIO()
    img2.save(img2_bytes, format='PNG')
    
    # Mock two different render results
    def render_side_effect(url, wait_time=3000):
        if "url1" in url or "example1" in url:
            return {
                "screenshot": base64.b64encode(img1_bytes.getvalue()).decode(),
                "dom": "<html>Page 1</html>",
                "status_code": 200,
                "metrics": {},
                "url": url
            }
        else:
            return {
                "screenshot": base64.b64encode(img2_bytes.getvalue()).decode(),
                "dom": "<html>Page 2</html>",
                "status_code": 200,
                "metrics": {},
                "url": url
            }
    
    mock_page_renderer.render_page.side_effect = render_side_effect
    
    response = client.post(
        "/api/v1/compare-visual",
        json={
            "url1": "https://example1.com",
            "url2": "https://example2.com"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "similarity_score" in data
    assert "hamming_distance" in data
    assert "is_similar" in data
    assert isinstance(data["similarity_score"], (int, float))
    assert 0 <= data["similarity_score"] <= 1

def test_compare_visual_error(client, mock_page_renderer):
    """Test compare-visual endpoint with rendering error"""
    mock_page_renderer.render_page.return_value = {
        "error": "Failed to render",
        "url": "https://invalid.com"
    }
    
    response = client.post(
        "/api/v1/compare-visual",
        json={
            "url1": "https://invalid.com",
            "url2": "https://example.com"
        }
    )
    
    assert response.status_code == 500

def test_analyze_dom(client, sample_html):
    """Test analyze-dom endpoint"""
    response = client.post(
        "/api/v1/analyze-dom",
        json={"html_content": sample_html}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "dom_hash" in data
    assert "element_count" in data
    assert "forms" in data
    assert "links" in data
    assert "images" in data
    assert "structure" in data
    assert "is_suspicious" in data
    
    # Verify forms were extracted
    assert isinstance(data["forms"], list)
    if len(data["forms"]) > 0:
        form = data["forms"][0]
        assert "action" in form
        assert "method" in form
        assert "fields" in form

def test_analyze_dom_empty_html(client):
    """Test analyze-dom endpoint with empty HTML"""
    response = client.post(
        "/api/v1/analyze-dom",
        json={"html_content": ""}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "dom_hash" in data
    assert data["element_count"] >= 0

def test_analyze_dom_invalid_request(client):
    """Test analyze-dom endpoint with invalid request"""
    response = client.post(
        "/api/v1/analyze-dom",
        json={}  # Missing html_content
    )
    
    assert response.status_code == 422  # Validation error

def test_analyze_page_brand_prediction(client, mock_page_renderer, mock_cnn_model, sample_html):
    """Test analyze-page endpoint with brand prediction"""
    mock_page_renderer.render_page.return_value = {
        "screenshot": base64.b64encode(b"fake_screenshot").decode(),
        "dom": sample_html,
        "status_code": 200,
        "metrics": {},
        "url": "https://example.com"
    }
    
    # Patch the cnn_model in routes
    with patch('src.api.routes.cnn_model', mock_cnn_model):
        response = client.post(
            "/api/v1/analyze-page",
            json={"url": "https://example.com"}
        )
        
        assert response.status_code == 200
        data = response.json()
        # Brand prediction may be None if model fails, but structure should be there
        assert "brand_prediction" in data

def test_analyze_page_s3_upload(client, mock_page_renderer, mock_s3_uploader, sample_html):
    """Test analyze-page endpoint with S3 upload"""
    mock_page_renderer.render_page.return_value = {
        "screenshot": base64.b64encode(b"fake_screenshot").decode(),
        "dom": sample_html,
        "status_code": 200,
        "metrics": {},
        "url": "https://example.com"
    }
    
    mock_s3_uploader.upload_screenshot.return_value = "https://s3.amazonaws.com/bucket/screenshots/test.png"
    
    with patch('src.api.routes.s3_uploader', mock_s3_uploader):
        response = client.post(
            "/api/v1/analyze-page",
            json={"url": "https://example.com"}
        )
        
        assert response.status_code == 200
        data = response.json()
        # Screenshot URL may be None if S3 upload fails, but should be present if successful
        assert "screenshot_url" in data
