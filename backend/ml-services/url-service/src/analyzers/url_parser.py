from urllib.parse import urlparse, parse_qs, unquote
from typing import Dict, List, Optional
import tldextract
import hashlib
import re

class URLParser:
    def __init__(self):
        self.extractor = tldextract.TLDExtract(cache_dir=False)
    
    def parse(self, url: str) -> Dict:
        """Parse URL into components"""
        try:
            parsed = urlparse(url)
            extracted = self.extractor(url)
            
            # Normalize URL
            normalized_url = self._normalize(url)
            
            # Generate hash
            url_hash = hashlib.sha256(normalized_url.encode()).hexdigest()
            
            # Parse query parameters
            query_params = parse_qs(parsed.query) if parsed.query else {}
            
            return {
                "original_url": url,
                "normalized_url": normalized_url,
                "url_hash": url_hash,
                "scheme": parsed.scheme,
                "domain": extracted.domain,
                "subdomain": extracted.subdomain,
                "registered_domain": f"{extracted.domain}.{extracted.suffix}",
                "tld": extracted.suffix,
                "path": parsed.path,
                "query": parsed.query,
                "query_params": {k: v[0] if len(v) == 1 else v for k, v in query_params.items()},
                "fragment": parsed.fragment,
                "port": parsed.port,
                "netloc": parsed.netloc
            }
        except Exception as e:
            raise ValueError(f"Failed to parse URL: {e}")
    
    def _normalize(self, url: str) -> str:
        """Normalize URL for consistent hashing"""
        # Remove default ports
        url = re.sub(r':80/', '/', url)
        url = re.sub(r':443/', '/', url)
        
        # Remove trailing slash (unless it's the root)
        if url.endswith('/') and len(urlparse(url).path) > 1:
            url = url[:-1]
        
        # Decode URL encoding
        url = unquote(url)
        
        # Lowercase
        url = url.lower()
        
        return url
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
