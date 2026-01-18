"""
Shared logging utilities for ML services
"""
import logging
import os
from typing import Optional

def setup_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Setup logger for ML services
    """
    logger = logging.getLogger(name)
    
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
