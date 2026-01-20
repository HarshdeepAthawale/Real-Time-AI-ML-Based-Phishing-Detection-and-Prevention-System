from bs4 import BeautifulSoup
from typing import Dict, List
import hashlib
import json

class DOMAnalyzer:
    def analyze(self, html_content: str) -> Dict:
        """Analyze DOM structure"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract form fields
        forms = self._extract_forms(soup)
        
        # Extract links
        links = self._extract_links(soup)
        
        # Extract images
        images = self._extract_images(soup)
        
        # Calculate DOM tree hash
        dom_hash = self._calculate_dom_hash(soup)
        
        # Analyze structure
        structure = self._analyze_structure(soup)
        
        return {
            "dom_hash": dom_hash,
            "element_count": len(soup.find_all()),
            "forms": forms,
            "links": links,
            "images": images,
            "structure": structure,
            "is_suspicious": self._is_suspicious(forms, links)
        }
    
    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract form fields"""
        forms = []
        for form in soup.find_all('form'):
            form_data = {
                "action": form.get('action', ''),
                "method": form.get('method', 'GET').upper(),
                "fields": []
            }
            
            for input_field in form.find_all(['input', 'textarea', 'select']):
                field_data = {
                    "type": input_field.get('type', 'text'),
                    "name": input_field.get('name', ''),
                    "placeholder": input_field.get('placeholder', ''),
                    "required": input_field.has_attr('required')
                }
                form_data["fields"].append(field_data)
            
            forms.append(form_data)
        return forms
    
    def _extract_links(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all links"""
        links = []
        for link in soup.find_all('a', href=True):
            links.append({
                "href": link['href'],
                "text": link.get_text(strip=True),
                "is_external": link['href'].startswith('http')
            })
        return links
    
    def _extract_images(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract images"""
        images = []
        for img in soup.find_all('img'):
            images.append({
                "src": img.get('src', ''),
                "alt": img.get('alt', ''),
                "width": img.get('width', ''),
                "height": img.get('height', '')
            })
        return images
    
    def _calculate_dom_hash(self, soup: BeautifulSoup) -> str:
        """Calculate hash of DOM structure (ignoring content)"""
        # Remove text content, keep only structure
        for element in soup.find_all():
            element.string = None
        
        dom_string = str(soup)
        return hashlib.sha256(dom_string.encode()).hexdigest()
    
    def _analyze_structure(self, soup: BeautifulSoup) -> Dict:
        """Analyze DOM structure patterns"""
        return {
            "has_login_form": bool(soup.find('form', {'action': lambda x: x and ('login' in x.lower() or 'signin' in x.lower())})),
            "has_password_field": bool(soup.find('input', {'type': 'password'})),
            "has_email_field": bool(soup.find('input', {'type': 'email'})),
            "has_credit_card_field": bool(soup.find('input', {'name': lambda x: x and ('card' in x.lower() or 'cvv' in x.lower())})),
            "iframe_count": len(soup.find_all('iframe')),
            "script_count": len(soup.find_all('script')),
            "external_script_count": len([s for s in soup.find_all('script', src=True) if s['src'].startswith('http')])
        }
    
    def _is_suspicious(self, forms: List[Dict], links: List[Dict]) -> bool:
        """Detect suspicious patterns"""
        # Multiple forms on same page
        if len(forms) > 3:
            return True
        
        # Excessive external links
        external_links = [l for l in links if l['is_external']]
        if len(external_links) > 20:
            return True
        
        return False
