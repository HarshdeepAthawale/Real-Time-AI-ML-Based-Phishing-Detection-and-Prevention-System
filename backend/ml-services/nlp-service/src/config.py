"""Configuration management for NLP service"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    service_name: str = "nlp-service"
    service_version: str = "1.0.0"
    port: int = 8000
    
    # Model paths
    model_dir: str = os.getenv("MODEL_DIR", "/app/models")
    phishing_model_path: str = os.path.join(model_dir, "phishing-detector")
    ai_detector_model_path: str = os.path.join(model_dir, "ai-detector")
    
    # Inference settings
    inference_device: str = os.getenv("INFERENCE_DEVICE", "cpu")
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "512"))
    
    # Redis configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl: int = 3600  # 1 hour
    
    # MongoDB configuration
    mongodb_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongodb_db: str = "phishing_detection"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "info")
    
    class Config:
        env_file = ".env"


settings = Settings()
