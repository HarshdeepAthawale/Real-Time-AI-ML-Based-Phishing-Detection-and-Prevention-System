"""
Visual Analysis Service - Phishing Detection using CNN models for image analysis
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict
import os
import time
from io import BytesIO
from dotenv import load_dotenv

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

load_dotenv()

app = FastAPI(title="Visual Phishing Detection Service", version="1.0.0")


class VisualAnalysisResponse(BaseModel):
    is_phishing: bool
    confidence: float
    features: Dict
    model_version: str
    detected_elements: Optional[list] = None
    processing_time_ms: Optional[float] = None


def analyze_image_basic(image_data: bytes) -> Dict:
    """Basic image analysis (rule-based until ML models are available)"""
    features = {}
    
    if not PIL_AVAILABLE:
        return {
            "error": "PIL not available",
            "image_size": len(image_data),
            "analysis_available": False
        }
    
    try:
        image = Image.open(BytesIO(image_data))
        
        # Basic image properties
        width, height = image.size
        format_type = image.format
        mode = image.mode
        
        # Convert to RGB if needed
        if mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate basic statistics
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)
        
        # Check for suspicious characteristics
        # Very bright images (potential overlay)
        is_very_bright = mean_brightness > 200
        
        # Low contrast (potential obfuscation)
        is_low_contrast = std_brightness < 30
        
        # Check image dimensions (very small images are suspicious)
        is_small = width < 100 or height < 100
        
        # Check aspect ratio (suspicious if very wide or tall)
        aspect_ratio = width / height if height > 0 else 1
        is_suspicious_aspect = aspect_ratio > 5 or aspect_ratio < 0.2
        
        features = {
            "width": width,
            "height": height,
            "format": format_type,
            "mode": mode,
            "mean_brightness": float(mean_brightness),
            "std_brightness": float(std_brightness),
            "is_very_bright": is_very_bright,
            "is_low_contrast": is_low_contrast,
            "is_small": is_small,
            "aspect_ratio": float(aspect_ratio),
            "is_suspicious_aspect": is_suspicious_aspect,
            "analysis_available": True
        }
    except Exception as e:
        features = {
            "error": str(e),
            "image_size": len(image_data),
            "analysis_available": False
        }
    
    return features


def calculate_phishing_score(features: Dict) -> float:
    """Calculate phishing probability based on image features"""
    if not features.get("analysis_available", False):
        return 0.3  # Default low score if analysis unavailable
    
    score = 0.0
    
    # Very bright image (potential overlay)
    if features.get("is_very_bright", False):
        score += 0.2
    
    # Low contrast (potential obfuscation)
    if features.get("is_low_contrast", False):
        score += 0.15
    
    # Very small image
    if features.get("is_small", False):
        score += 0.1
    
    # Suspicious aspect ratio
    if features.get("is_suspicious_aspect", False):
        score += 0.15
    
    return min(score, 1.0)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "visual-service"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "visual-service", "version": "1.0.0"}


@app.post("/analyze", response_model=VisualAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze image content for phishing indicators using CNN models
    """
    start_time = time.time()
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Analyze image
        features = analyze_image_basic(image_data)
        
        # Calculate phishing probability
        phishing_probability = calculate_phishing_score(features)
        
        # Determine if phishing (threshold: 0.4 for images - more lenient)
        is_phishing = phishing_probability >= 0.4
        
        processing_time = (time.time() - start_time) * 1000
        
        return VisualAnalysisResponse(
            is_phishing=is_phishing,
            confidence=phishing_probability,
            features=features,
            model_version="1.0.0-rule-based",
            detected_elements=None,
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
