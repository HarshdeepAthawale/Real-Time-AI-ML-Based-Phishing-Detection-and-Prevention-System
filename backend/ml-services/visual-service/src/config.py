"""Configuration for Visual service"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    service_name: str = "visual-service"
    service_version: str = "1.0.0"
    port: int = 8002
    
    # Model paths
    model_dir: str = os.getenv("MODEL_DIR", "/app/models")
    cnn_model_path: str = os.path.join(model_dir, "brand-classifier")
    brand_db_path: str = os.path.join(model_dir, "brand-database")
    
    # Inference settings
    inference_device: str = os.getenv("INFERENCE_DEVICE", "cpu")
    
    # Redis configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl: int = 3600  # 1 hour
    
    # Browser settings
    browser_timeout: int = int(os.getenv("BROWSER_TIMEOUT", "30000"))  # 30 seconds
    screenshot_quality: int = int(os.getenv("SCREENSHOT_QUALITY", "80"))
    
    # S3 configuration
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    aws_region: str = os.getenv("AWS_REGION", "ap-south-1")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "info")
    
    class Config:
        env_file = ".env"


settings = Settings()
