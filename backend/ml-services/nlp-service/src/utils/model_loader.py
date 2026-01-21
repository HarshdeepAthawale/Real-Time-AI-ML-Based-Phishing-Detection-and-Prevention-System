import os
import logging
from src.models.phishing_classifier import PhishingClassifier
from src.models.ai_detector import AIGeneratedDetector

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.phishing_classifier = None
            self.ai_detector = None
            self._models_loaded = False
            self._initialized = True
    
    async def load_all_models(self):
        """Load all models"""
        try:
            # Get configuration from environment
            model_dir = os.getenv("MODEL_DIR", "./models")
            device = os.getenv("DEVICE", "cpu")
            
            # Check for GPU availability
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            
            logger.info(f"Loading models from {model_dir} on device: {device}")
            
            # Load phishing classifier
            # Check multiple possible paths
            phishing_model_paths = [
                os.path.join(model_dir, "phishing-bert-v1"),
                os.path.join(model_dir, "phishing-roberta-v1"),
                os.getenv("PHISHING_MODEL_PATH"),
            ]
            
            phishing_model_path = None
            for path in phishing_model_paths:
                if path and os.path.exists(path):
                    phishing_model_path = path
                    logger.info(f"Found phishing model at: {path}")
                    break
            
            if not phishing_model_path:
                logger.info("No custom phishing model found. Using pre-trained RoBERTa-base model.")
                logger.info("To train a custom model, run: python training/train_phishing_model.py")
            
            self.phishing_classifier = PhishingClassifier(
                model_path=phishing_model_path,
                device=device
            )
            
            # Load AI detector
            # Check multiple possible paths
            ai_model_paths = [
                os.path.join(model_dir, "ai-detector-v1"),
                os.path.join(model_dir, "ai-detector-roberta-v1"),
                os.getenv("AI_DETECTOR_MODEL_PATH"),
            ]
            
            ai_model_path = None
            for path in ai_model_paths:
                if path and os.path.exists(path):
                    ai_model_path = path
                    logger.info(f"Found AI detector model at: {path}")
                    break
            
            if not ai_model_path:
                logger.info("No custom AI detector model found. Using pre-trained RoBERTa-base model.")
                logger.info("To train a custom model, run: python training/train_ai_detector.py")
            
            self.ai_detector = AIGeneratedDetector(
                model_path=ai_model_path,
                device=device
            )
            
            # Verify models are loaded
            if self.phishing_classifier.model is None:
                logger.warning("Phishing classifier model is None - will use rule-based fallback")
            if self.ai_detector.model is None:
                logger.warning("AI detector model is None - will use heuristic fallback")
            
            self._models_loaded = True
            logger.info("All models loaded successfully")
            logger.info(f"Phishing classifier ready: {self.phishing_classifier.model is not None}")
            logger.info(f"AI detector ready: {self.ai_detector.model is not None}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            self._models_loaded = False
            # Don't raise - allow service to start with fallback methods
            logger.warning("Service will continue with rule-based fallback methods")
    
    async def unload_all_models(self):
        """Unload models to free memory"""
        try:
            self.phishing_classifier = None
            self.ai_detector = None
            self._models_loaded = False
            logger.info("Models unloaded")
        except Exception as e:
            logger.error(f"Error unloading models: {e}")
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._models_loaded and self.phishing_classifier is not None and self.ai_detector is not None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        return cls._instance
