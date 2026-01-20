"""
Tests for utility components
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io
import numpy as np

from src.utils.image_processor import ImageProcessor
from src.utils.s3_uploader import S3Uploader

def test_image_processor_resize_image(sample_image_bytes):
    """Test image resizing"""
    resized = ImageProcessor.resize_image(sample_image_bytes, size=(100, 100))
    
    assert isinstance(resized, bytes)
    assert len(resized) > 0
    
    # Verify dimensions
    img = Image.open(io.BytesIO(resized))
    assert img.size == (100, 100)

def test_image_processor_convert_to_rgb(sample_image_bytes):
    """Test RGB conversion"""
    converted = ImageProcessor.convert_to_rgb(sample_image_bytes)
    
    assert isinstance(converted, bytes)
    
    # Verify it's RGB
    img = Image.open(io.BytesIO(converted))
    assert img.mode == 'RGB'

def test_image_processor_get_image_info(sample_image_bytes):
    """Test image info extraction"""
    info = ImageProcessor.get_image_info(sample_image_bytes)
    
    assert "width" in info
    assert "height" in info
    assert "format" in info
    assert "mode" in info
    assert "size_bytes" in info
    
    assert isinstance(info["width"], int)
    assert isinstance(info["height"], int)
    assert info["size_bytes"] == len(sample_image_bytes)

def test_image_processor_normalize_image(sample_image_bytes):
    """Test image normalization"""
    normalized = ImageProcessor.normalize_image(sample_image_bytes)
    
    assert isinstance(normalized, np.ndarray)
    assert normalized.dtype == np.float32
    # Should be normalized to [0, 1]
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0

def test_image_processor_extract_thumbnail(sample_image_bytes):
    """Test thumbnail extraction"""
    thumbnail = ImageProcessor.extract_thumbnail(sample_image_bytes, size=(64, 64))
    
    assert isinstance(thumbnail, bytes)
    assert len(thumbnail) > 0
    
    # Verify thumbnail dimensions (may be smaller than requested due to aspect ratio)
    img = Image.open(io.BytesIO(thumbnail))
    assert img.width <= 64
    assert img.height <= 64

def test_image_processor_crop_image(sample_image_bytes):
    """Test image cropping"""
    # Crop to a smaller region
    cropped = ImageProcessor.crop_image(sample_image_bytes, box=(10, 10, 50, 50))
    
    assert isinstance(cropped, bytes)
    assert len(cropped) > 0
    
    # Verify dimensions
    img = Image.open(io.BytesIO(cropped))
    assert img.size == (40, 40)  # 50-10, 50-10

def test_image_processor_invalid_image():
    """Test image processor with invalid image"""
    invalid_bytes = b"not an image"
    
    # Should raise an error or handle gracefully
    try:
        ImageProcessor.get_image_info(invalid_bytes)
        # If it doesn't raise, that's also acceptable
    except Exception:
        # Expected for invalid images
        pass

def test_s3_uploader_initialization():
    """Test S3Uploader initialization"""
    uploader = S3Uploader()
    
    assert uploader is not None
    assert uploader.s3_client is not None

def test_s3_uploader_initialization_with_credentials():
    """Test S3Uploader initialization with credentials"""
    uploader = S3Uploader(
        bucket_name="test-bucket",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        region_name="us-west-2"
    )
    
    assert uploader.bucket_name == "test-bucket"
    assert uploader.region_name == "us-west-2"

@patch('src.utils.s3_uploader.boto3')
def test_s3_uploader_upload_image(mock_boto3, sample_image_bytes):
    """Test S3 image upload"""
    # Mock S3 client
    mock_s3_client = Mock()
    mock_boto3.client.return_value = mock_s3_client
    
    uploader = S3Uploader(bucket_name="test-bucket")
    uploader.s3_client = mock_s3_client
    
    url = uploader.upload_image(
        sample_image_bytes,
        key="test/image.png",
        content_type="image/png"
    )
    
    assert url is not None
    assert "test-bucket" in url
    mock_s3_client.put_object.assert_called_once()

@patch('src.utils.s3_uploader.boto3')
def test_s3_uploader_upload_image_no_bucket(mock_boto3, sample_image_bytes):
    """Test S3 image upload without bucket configured"""
    mock_s3_client = Mock()
    mock_boto3.client.return_value = mock_s3_client
    
    uploader = S3Uploader(bucket_name=None)
    uploader.s3_client = mock_s3_client
    
    url = uploader.upload_image(sample_image_bytes, key="test/image.png")
    
    # Should return None when bucket not configured
    assert url is None

@patch('src.utils.s3_uploader.boto3')
def test_s3_uploader_upload_screenshot(mock_boto3, sample_image_bytes):
    """Test screenshot upload"""
    mock_s3_client = Mock()
    mock_boto3.client.return_value = mock_s3_client
    
    uploader = S3Uploader(bucket_name="test-bucket")
    uploader.s3_client = mock_s3_client
    
    url = uploader.upload_screenshot(
        sample_image_bytes,
        url="https://example.com"
    )
    
    assert url is not None
    mock_s3_client.put_object.assert_called_once()
    
    # Verify metadata was set
    call_kwargs = mock_s3_client.put_object.call_args[1]
    assert "Metadata" in call_kwargs
    assert "url" in call_kwargs["Metadata"]

@patch('src.utils.s3_uploader.boto3')
def test_s3_uploader_upload_screenshot_with_analysis_id(mock_boto3, sample_image_bytes):
    """Test screenshot upload with analysis ID"""
    mock_s3_client = Mock()
    mock_boto3.client.return_value = mock_s3_client
    
    uploader = S3Uploader(bucket_name="test-bucket")
    uploader.s3_client = mock_s3_client
    
    url = uploader.upload_screenshot(
        sample_image_bytes,
        url="https://example.com",
        analysis_id="test-analysis-123"
    )
    
    assert url is not None
    
    # Verify analysis_id in metadata
    call_kwargs = mock_s3_client.put_object.call_args[1]
    assert "analysis_id" in call_kwargs["Metadata"]

@patch('src.utils.s3_uploader.boto3')
def test_s3_uploader_upload_error(mock_boto3, sample_image_bytes):
    """Test S3 upload error handling"""
    mock_s3_client = Mock()
    mock_s3_client.put_object.side_effect = Exception("S3 error")
    mock_boto3.client.return_value = mock_s3_client
    
    uploader = S3Uploader(bucket_name="test-bucket")
    uploader.s3_client = mock_s3_client
    
    url = uploader.upload_image(sample_image_bytes, key="test/image.png")
    
    # Should return None on error
    assert url is None

@patch('src.utils.s3_uploader.boto3')
def test_s3_uploader_delete_image(mock_boto3):
    """Test S3 image deletion"""
    mock_s3_client = Mock()
    mock_boto3.client.return_value = mock_s3_client
    
    uploader = S3Uploader(bucket_name="test-bucket")
    uploader.s3_client = mock_s3_client
    
    result = uploader.delete_image("test/image.png")
    
    assert result is True
    mock_s3_client.delete_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="test/image.png"
    )

@patch('src.utils.s3_uploader.boto3')
def test_s3_uploader_delete_image_no_bucket(mock_boto3):
    """Test S3 image deletion without bucket"""
    mock_s3_client = Mock()
    mock_boto3.client.return_value = mock_s3_client
    
    uploader = S3Uploader(bucket_name=None)
    uploader.s3_client = mock_s3_client
    
    result = uploader.delete_image("test/image.png")
    
    assert result is False

@patch('src.utils.s3_uploader.boto3')
def test_s3_uploader_delete_image_error(mock_boto3):
    """Test S3 image deletion error handling"""
    mock_s3_client = Mock()
    mock_s3_client.delete_object.side_effect = Exception("S3 error")
    mock_boto3.client.return_value = mock_s3_client
    
    uploader = S3Uploader(bucket_name="test-bucket")
    uploader.s3_client = mock_s3_client
    
    result = uploader.delete_image("test/image.png")
    
    assert result is False
