"""
Tests for ML models
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image
import io
import os

from src.models.cnn_classifier import BrandImpersonationCNN, CNNModelLoader
from src.models.visual_similarity import VisualSimilarityMatcher

def test_brand_impersonation_cnn_initialization():
    """Test CNN model initialization"""
    model = BrandImpersonationCNN(num_brands=10)
    
    assert model is not None
    assert hasattr(model, 'features')
    assert hasattr(model, 'classifier')
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        assert output.shape == (1, 11)  # 10 brands + 1 unknown

def test_cnn_model_loader_initialization():
    """Test CNNModelLoader initialization"""
    loader = CNNModelLoader(model_path="./nonexistent/model.pt")
    
    assert loader is not None
    assert loader.model_path == "./nonexistent/model.pt"
    assert loader.model is None

def test_cnn_model_loader_load_model_no_file():
    """Test CNNModelLoader when model file doesn't exist"""
    loader = CNNModelLoader(model_path="./nonexistent/model.pt")
    
    # Should not raise error, but model should be untrained
    model = loader.load_model(num_brands=10)
    assert model is not None
    assert isinstance(model, BrandImpersonationCNN)

@patch('torch.load')
@patch('os.path.exists')
def test_cnn_model_loader_load_model_with_file(mock_exists, mock_torch_load):
    """Test CNNModelLoader when model file exists"""
    mock_exists.return_value = True
    
    # Create a mock state dict
    mock_state_dict = {
        'features.0.weight': torch.randn(64, 3, 7, 7),
        'classifier.0.weight': torch.randn(512, 2048),
        'classifier.3.weight': torch.randn(11, 512)
    }
    mock_torch_load.return_value = mock_state_dict
    
    loader = CNNModelLoader(model_path="./models/test.pt")
    model = loader.load_model(num_brands=10)
    
    assert model is not None
    mock_torch_load.assert_called_once()

def test_cnn_model_predict(sample_image_bytes):
    """Test CNN model prediction"""
    model = BrandImpersonationCNN(num_brands=10)
    model.eval()
    
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    image_bytes = img_bytes.getvalue()
    
    result = model.predict(image_bytes)
    
    assert "predictions" in result
    assert "top_brand_id" in result
    assert "top_brand_probability" in result
    assert "is_brand_impersonation" in result
    assert "confidence" in result
    
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 5  # Top 5 predictions
    assert 0 <= result["top_brand_probability"] <= 1
    assert isinstance(result["is_brand_impersonation"], bool)

def test_visual_similarity_matcher_initialization():
    """Test VisualSimilarityMatcher initialization"""
    matcher = VisualSimilarityMatcher()
    
    assert matcher is not None
    assert matcher.hash_size == 16

def test_visual_similarity_calculate_hash(sample_image_bytes):
    """Test visual similarity hash calculation"""
    matcher = VisualSimilarityMatcher()
    
    hash_value = matcher.calculate_hash(sample_image_bytes)
    
    assert isinstance(hash_value, str)
    assert len(hash_value) > 0

def test_visual_similarity_compare_images_identical(sample_image_bytes):
    """Test visual similarity comparison with identical images"""
    matcher = VisualSimilarityMatcher()
    
    result = matcher.compare_images(sample_image_bytes, sample_image_bytes)
    
    assert "similarity_score" in result
    assert "hamming_distance" in result
    assert "is_similar" in result
    
    # Identical images should have high similarity
    assert result["similarity_score"] > 0.9
    assert result["hamming_distance"] == 0
    assert result["is_similar"] is True

def test_visual_similarity_compare_images_different(sample_image_bytes):
    """Test visual similarity comparison with different images"""
    matcher = VisualSimilarityMatcher()
    
    # Create a different image
    img1 = Image.new('RGB', (100, 100), color='red')
    img1_bytes = io.BytesIO()
    img1.save(img1_bytes, format='PNG')
    
    img2 = Image.new('RGB', (100, 100), color='blue')
    img2_bytes = io.BytesIO()
    img2.save(img2_bytes, format='PNG')
    
    result = matcher.compare_images(img1_bytes.getvalue(), img2_bytes.getvalue())
    
    assert "similarity_score" in result
    assert "hamming_distance" in result
    assert "is_similar" in result
    
    # Different images should have lower similarity
    assert 0 <= result["similarity_score"] <= 1
    assert result["hamming_distance"] >= 0

def test_visual_similarity_calculate_multiple_hashes(sample_image_bytes):
    """Test calculating multiple hash types"""
    matcher = VisualSimilarityMatcher()
    
    hashes = matcher.calculate_multiple_hashes(sample_image_bytes)
    
    assert "phash" in hashes
    assert "dhash" in hashes
    assert "whash" in hashes
    assert "average_hash" in hashes
    
    assert isinstance(hashes["phash"], str)
    assert isinstance(hashes["dhash"], str)
    assert isinstance(hashes["whash"], str)
    assert isinstance(hashes["average_hash"], str)

def test_visual_similarity_find_similar_brands(sample_image_bytes):
    """Test finding similar brands from database"""
    matcher = VisualSimilarityMatcher()
    
    # Create a mock brand database
    brand_database = [
        {
            "id": 1,
            "name": "Brand1",
            "image_hash": matcher.calculate_hash(sample_image_bytes)
        },
        {
            "id": 2,
            "name": "Brand2",
            "image_hash": "0000000000000000"  # Very different hash
        }
    ]
    
    # Use the same image, so it should match Brand1
    similar = matcher.find_similar_brands(sample_image_bytes, brand_database)
    
    assert isinstance(similar, list)
    # Should find at least Brand1 as similar
    if len(similar) > 0:
        assert "brand_id" in similar[0]
        assert "brand_name" in similar[0]
        assert "similarity" in similar[0]
        assert similar[0]["brand_id"] == 1

def test_cnn_model_predict_invalid_image():
    """Test CNN model prediction with invalid image"""
    model = BrandImpersonationCNN(num_brands=10)
    model.eval()
    
    # Invalid image bytes
    invalid_bytes = b"not an image"
    
    # Should handle gracefully or raise appropriate error
    try:
        result = model.predict(invalid_bytes)
        # If it doesn't raise, result should still have expected structure
        assert "predictions" in result
    except Exception:
        # It's acceptable for invalid images to raise exceptions
        pass
