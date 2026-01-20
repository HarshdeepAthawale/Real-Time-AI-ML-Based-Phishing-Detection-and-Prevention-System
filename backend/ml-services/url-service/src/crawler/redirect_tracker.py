import requests
from typing import List, Dict
from urllib.parse import urljoin, urlparse

class RedirectTracker:
    def __init__(self, max_hops: int = 10):
        self.max_hops = max_hops
        self.session = requests.Session()
        self.session.max_redirects = 0  # Handle redirects manually
    
    def track(self, url: str) -> Dict:
        """Track redirect chain"""
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
                response = self.session.get(current_url, allow_redirects=False, timeout=5)
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
                
            except Exception as e:
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
            "redirect_count": len([r for r in redirects if r.get("redirect_type")]),
            "redirects": redirects,
            "is_suspicious": self._is_suspicious(redirects)
        }
    
    def _is_suspicious(self, redirects: List[Dict]) -> bool:
        """Detect suspicious redirect patterns"""
        # Too many redirects
        if len(redirects) > 5:
            return True
        
        # Redirects to different domains
        domains = set()
        for redirect in redirects:
            domain = urlparse(redirect["url"]).netloc
            if domain:
                domains.add(domain)
        
        if len(domains) > 3:
            return True
        
        return False
