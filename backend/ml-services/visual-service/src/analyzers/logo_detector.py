"""Logo detection using template matching"""
import cv2
import numpy as np
from typing import List, Dict
from src.utils.logger import logger


class LogoDetector:
    """Detect logos in screenshots"""
    
    def __init__(self):
        # Logo templates would be loaded from database/files
        self.logo_templates = {}
    
    def detect(self, image_bytes: bytes) -> Dict:
        """
        Detect logos in screenshot (simplified version)
        
        Args:
            image_bytes: Screenshot image bytes
            
        Returns:
            Dictionary with detection results
        """
        logos = self.detect_logos(image_bytes)
        return {
            "logos_detected": len(logos),
            "logos": logos,
            "has_brand_logo": len(logos) > 0
        }
    
    def detect_logos(self, image_bytes: bytes) -> List[Dict]:
        """
        Detect logos in screenshot
        
        Args:
            image_bytes: Screenshot image bytes
            
        Returns:
            List of detected logos with positions
        """
        if not self.logo_templates:
            logger.warning("No logo templates loaded")
            return []
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
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
        
        except Exception as e:
            logger.error(f"Error detecting logos: {e}")
            return []
    
    def load_templates(self, template_paths: Dict[str, str]):
        """Load logo templates from files"""
        for logo_id, path in template_paths.items():
            try:
                template = cv2.imread(path)
                if template is not None:
                    self.logo_templates[logo_id] = template
                    logger.info(f"Loaded template for {logo_id}")
            except Exception as e:
                logger.error(f"Error loading template {logo_id}: {e}")
