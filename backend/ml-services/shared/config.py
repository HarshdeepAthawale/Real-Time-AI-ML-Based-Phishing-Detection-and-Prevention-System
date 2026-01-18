"""
Shared configuration utilities for ML services
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class MLServiceSettings(BaseSettings):
    """Base settings for ML services"""
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    mongodb_url: Optional[str] = os.getenv("MONGODB_URL")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    aws_region: str = os.getenv("AWS_REGION", "ap-south-1")
    s3_bucket_models: str = os.getenv("S3_BUCKET_MODELS", "")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
