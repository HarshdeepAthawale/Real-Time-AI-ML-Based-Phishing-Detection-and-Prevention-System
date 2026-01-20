import cv2
import numpy as np
from typing import Dict, List, Optional

class LogoDetector:
    def __init__(self):
        # Load pre-trained logo detection model (YOLO or similar)
        # For now, use template matching
        self.logo_templates = {}  # Would be loaded from database
    
    def detect_logos(self, image_bytes: bytes) -> List[Dict]:
        """Detect logos in screenshot"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return []
        
        detected_logos = []
        
        # Template matching for known logos
        for logo_id, template in self.logo_templates.items():
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.7)  # Threshold
            
            for pt in zip(*locations[::-1]):
                detected_logos.append({
                    "logo_id": logo_id,
                    "position": {"x": int(pt[0]), "y": int(pt[1])},
                    "confidence": float(result[pt[1], pt[0]])
                })
        
        return detected_logos
    
    def add_logo_template(self, logo_id: str, template_bytes: bytes):
        """Add a logo template for matching"""
        nparr = np.frombuffer(template_bytes, np.uint8)
        template = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if template is not None:
            self.logo_templates[logo_id] = template
    
    def detect_brand_colors(self, image_bytes: bytes) -> Dict:
        """Detect dominant colors that might indicate brand"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"colors": []}
        
        # Resize for faster processing
        image_small = cv2.resize(image, (100, 100))
        
        # Reshape to 1D array
        pixels = image_small.reshape(-1, 3)
        
        # Convert to float
        pixels = np.float32(pixels)
        
        # K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 5
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count occurrences
        unique, counts = np.unique(labels, return_counts=True)
        
        # Get dominant colors
        colors = []
        for i, count in zip(unique, counts):
            color = centers[i].astype(int).tolist()
            colors.append({
                "rgb": color,
                "frequency": float(count / len(pixels))
            })
        
        # Sort by frequency
        colors.sort(key=lambda x: x['frequency'], reverse=True)
        
        return {"colors": colors[:5]}  # Top 5 colors
