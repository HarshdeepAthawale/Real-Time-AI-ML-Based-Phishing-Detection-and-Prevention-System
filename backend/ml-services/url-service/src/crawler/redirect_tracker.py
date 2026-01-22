"""HTTP redirect chain tracking"""
import requests
from typing import List, Dict
from urllib.parse import urljoin
from src.config import settings
from src.utils.logger import logger


class RedirectTracker:
    """Track HTTP redirect chains"""
    
    def __init__(self):
        self.max_hops = settings.max_redirects
        self.session = requests.Session()
        self.session.max_redirects = 0  # Handle redirects manually
    
    def track(self, url: str) -> Dict:
        """
        Track redirect chain for URL
        
        Args:
            url: URL to track
            
        Returns:
            Dictionary with redirect chain analysis
        """
        redirects = []
        current_url = url
        visited = set()
        
        for hop in range(self.max_hops):
            if current_url in visited:
                redirects.append({
                    "url": current_url,
                    "status_code": 0,
                    "redirect_type": "loop",
                    "hop": hop
                })
                break
            
            visited.add(current_url)
            
            try:
                response = self.session.get(
                    current_url,
                    allow_redirects=False,
                    timeout=5,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                status_code = response.status_code
                
                redirect_type = None
                next_url = None
                
                if status_code in [301, 302, 303, 307, 308]:
                    redirect_type = "permanent" if status_code in [301, 308] else "temporary"
                    next_url = response.headers.get('Location')
                    
                    if next_url:
                        next_url = urljoin(current_url, next_url)
                
                redirects.append({
                    "url": current_url,
                    "status_code": status_code,
                    "redirect_type": redirect_type,
                    "next_url": next_url,
                    "hop": hop
                })
                
                if not next_url or status_code not in [301, 302, 303, 307, 308]:
                    break
                
                current_url = next_url
            
            except requests.exceptions.Timeout:
                redirects.append({
                    "url": current_url,
                    "status_code": 0,
                    "redirect_type": "timeout",
                    "hop": hop
                })
                break
            
            except Exception as e:
                logger.debug(f"Error tracking redirect for {current_url}: {e}")
                redirects.append({
                    "url": current_url,
                    "status_code": 0,
                    "redirect_type": "error",
                    "error": str(e),
                    "hop": hop
                })
                break
        
        return {
            "original_url": url,
            "final_url": redirects[-1]["url"] if redirects else url,
            "redirect_count": len([r for r in redirects if r.get("redirect_type") and r["redirect_type"] not in ["error", "timeout", "loop"]]),
            "redirects": redirects,
            "is_suspicious": self._is_suspicious(redirects),
            "has_loop": any(r.get("redirect_type") == "loop" for r in redirects)
        }
    
    def _is_suspicious(self, redirects: List[Dict]) -> bool:
        """Detect suspicious redirect patterns"""
        # Too many redirects
        if len(redirects) > 5:
            return True
        
        # Redirects to multiple different domains
        domains = set()
        for redirect in redirects:
            from urllib.parse import urlparse
            try:
                domain = urlparse(redirect["url"]).netloc
                domains.add(domain)
            except Exception:
                pass
        
        if len(domains) > 3:
            return True
        
        return False
