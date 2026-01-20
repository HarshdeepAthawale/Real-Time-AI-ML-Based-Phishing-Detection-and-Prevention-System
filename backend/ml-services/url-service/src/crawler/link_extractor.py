import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse
import re

class LinkExtractor:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_links(self, url: str) -> Dict:
        """Extract all links from a webpage"""
        results = {
            "url": url,
            "links": [],
            "external_links": [],
            "internal_links": [],
            "error": None
        }
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            base_domain = urlparse(url).netloc
            
            # Extract all anchor tags
            links = []
            for tag in soup.find_all('a', href=True):
                href = tag.get('href')
                if href:
                    absolute_url = urljoin(url, href)
                    parsed = urlparse(absolute_url)
                    
                    link_info = {
                        "url": absolute_url,
                        "text": tag.get_text(strip=True),
                        "domain": parsed.netloc,
                        "is_external": parsed.netloc != base_domain if parsed.netloc else False
                    }
                    links.append(link_info)
                    
                    if link_info["is_external"]:
                        results["external_links"].append(link_info)
                    else:
                        results["internal_links"].append(link_info)
            
            results["links"] = links
            
        except requests.exceptions.RequestException as e:
            results["error"] = str(e)
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def extract_domains(self, url: str) -> Set[str]:
        """Extract unique domains from page links"""
        link_data = self.extract_links(url)
        domains = set()
        
        for link in link_data.get("links", []):
            domain = link.get("domain")
            if domain:
                domains.add(domain)
        
        return domains
