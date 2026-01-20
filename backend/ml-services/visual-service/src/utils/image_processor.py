from PIL import Image
import io
import numpy as np
from typing import Dict, Tuple, Optional

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def resize_image(image_bytes: bytes, size: Tuple[int, int] = (224, 224)) -> bytes:
        """Resize image to specified dimensions"""
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize(size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @staticmethod
    def convert_to_rgb(image_bytes: bytes) -> bytes:
        """Convert image to RGB format"""
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @staticmethod
    def get_image_info(image_bytes: bytes) -> Dict:
        """Get image metadata"""
        image = Image.open(io.BytesIO(image_bytes))
        
        return {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "size_bytes": len(image_bytes)
        }
    
    @staticmethod
    def normalize_image(image_bytes: bytes) -> np.ndarray:
        """Normalize image for ML model input"""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image)
        
        # Normalize to [0, 1]
        normalized = image_array.astype(np.float32) / 255.0
        
        return normalized
    
    @staticmethod
    def extract_thumbnail(image_bytes: bytes, size: Tuple[int, int] = (128, 128)) -> bytes:
        """Extract thumbnail from image"""
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @staticmethod
    def crop_image(image_bytes: bytes, box: Tuple[int, int, int, int]) -> bytes:
        """Crop image to specified box (left, top, right, bottom)"""
        image = Image.open(io.BytesIO(image_bytes))
        cropped = image.crop(box)
        
        buffer = io.BytesIO()
        cropped.save(buffer, format='PNG')
        return buffer.getvalue()
