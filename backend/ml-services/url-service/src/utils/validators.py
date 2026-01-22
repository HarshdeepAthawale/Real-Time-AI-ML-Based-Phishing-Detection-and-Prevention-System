"""Input validation utilities"""
import re
from urllib.parse import urlparse
from typing import Optional


def is_valid_url(url: str) -> bool:
    """Check if string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_valid_domain(domain: str) -> bool:
    """Check if string is a valid domain"""
    domain_pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
    return bool(re.match(domain_pattern, domain))


def is_ip_address(address: str) -> bool:
    """Check if string is an IP address"""
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if not re.match(ip_pattern, address):
        return False
    
    # Validate octets
    octets = address.split('.')
    return all(0 <= int(octet) <= 255 for octet in octets)


def extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None
