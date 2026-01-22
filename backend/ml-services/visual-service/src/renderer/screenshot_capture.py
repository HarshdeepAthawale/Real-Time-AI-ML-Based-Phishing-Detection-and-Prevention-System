"""Screenshot capture utilities"""
from typing import Optional
import base64


def decode_screenshot(screenshot_base64: str) -> bytes:
    """
    Decode base64 screenshot to bytes
    
    Args:
        screenshot_base64: Base64 encoded screenshot
        
    Returns:
        Screenshot bytes
    """
    return base64.b64decode(screenshot_base64)


def encode_screenshot(screenshot_bytes: bytes) -> str:
    """
    Encode screenshot bytes to base64
    
    Args:
        screenshot_bytes: Screenshot bytes
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(screenshot_bytes).decode()
