"""Configuration for URL service"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    service_name: str = "url-service"
    service_version: str = "1.0.0"
    port: int = 8001
    
    # Model paths
    model_dir: str = os.getenv("MODEL_DIR", "/app/models")
    gnn_model_path: str = os.path.join(model_dir, "gnn-domain-classifier")
    
    # Inference settings
    inference_device: str = os.getenv("INFERENCE_DEVICE", "cpu")
    
    # Redis configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl: int = 7200  # 2 hours for URLs
    
    # Analysis settings
    dns_timeout: int = int(os.getenv("DNS_TIMEOUT", "5"))
    whois_timeout: int = int(os.getenv("WHOIS_TIMEOUT", "10"))
    max_redirects: int = int(os.getenv("MAX_REDIRECTS", "10"))
    enable_ssl_verify: bool = os.getenv("ENABLE_SSL_VERIFY", "true").lower() == "true"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "info")
    
    class Config:
        env_file = ".env"


settings = Settings()
