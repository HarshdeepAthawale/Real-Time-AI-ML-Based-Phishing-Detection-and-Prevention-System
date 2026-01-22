"""Image processing utilities"""
from PIL import Image
import io
from typing import Tuple


def resize_image(image_bytes: bytes, max_size: Tuple[int, int] = (1920, 1080)) -> bytes:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image_bytes: Original image bytes
        max_size: Maximum dimensions (width, height)
        
    Returns:
        Resized image bytes
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        output = io.BytesIO()
        image.save(output, format='PNG')
        return output.getvalue()
    
    except Exception as e:
        raise ValueError(f"Error resizing image: {e}")


def convert_to_jpeg(image_bytes: bytes, quality: int = 80) -> bytes:
    """
    Convert image to JPEG format
    
    Args:
        image_bytes: Original image bytes
        quality: JPEG quality (1-100)
        
    Returns:
        JPEG image bytes
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality)
        return output.getvalue()
    
    except Exception as e:
        raise ValueError(f"Error converting to JPEG: {e}")
