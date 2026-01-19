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
            model_dir = os.getenv("MODEL_DIR", "./models")
            device = os.getenv("DEVICE", "cpu")
            
            logger.info(f"Loading models from {model_dir} on device: {device}")
            
            # Load phishing classifier
            phishing_model_path = os.path.join(model_dir, "phishing-bert-v1")
            if not os.path.exists(phishing_model_path):
                logger.warning(f"Phishing model path {phishing_model_path} not found. Using default model.")
                phishing_model_path = None
            
            self.phishing_classifier = PhishingClassifier(
                model_path=phishing_model_path,
                device=device
            )
            
            # Load AI detector
            ai_model_path = os.path.join(model_dir, "ai-detector-v1")
            if not os.path.exists(ai_model_path):
                logger.warning(f"AI detector model path {ai_model_path} not found. Using default model.")
                ai_model_path = None
            
            self.ai_detector = AIGeneratedDetector(
                model_path=ai_model_path,
                device=device
            )
            
            self._models_loaded = True
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            self._models_loaded = False
            raise
    
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
