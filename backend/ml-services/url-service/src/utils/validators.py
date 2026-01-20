import re
from typing import Optional
from urllib.parse import urlparse

class URLValidator:
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def is_valid_domain(domain: str) -> bool:
        """Validate domain format"""
        domain_pattern = re.compile(
            r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
        )
        return bool(domain_pattern.match(domain))
    
    @staticmethod
    def sanitize_url(url: str) -> Optional[str]:
        """Sanitize and normalize URL"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme:
                url = f"https://{url}"
                parsed = urlparse(url)
            
            if not parsed.netloc:
                return None
            
            return url
        except:
            return None
