"""Extract links from web pages"""
import requests
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import urljoin
from src.utils.logger import logger


class LinkExtractor:
    """Extract links from web pages"""
    
    def extract(self, url: str, max_links: int = 100) -> List[str]:
        """
        Extract links from webpage
        
        Args:
            url: URL to extract links from
            max_links: Maximum number of links to extract
            
        Returns:
            List of extracted URLs
        """
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                links.append(absolute_url)
                
                if len(links) >= max_links:
                    break
            
            return links
        
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {e}")
            return []
