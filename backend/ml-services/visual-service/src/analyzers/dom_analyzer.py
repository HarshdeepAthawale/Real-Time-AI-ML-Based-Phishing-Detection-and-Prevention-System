"""DOM structure analysis"""
from bs4 import BeautifulSoup
from typing import Dict, List
import hashlib


class DOMAnalyzer:
    """Analyze DOM structure for suspicious patterns"""
    
    def analyze(self, html_content: str) -> Dict:
        """
        Analyze DOM structure
        
        Args:
            html_content: HTML content to analyze
            
        Returns:
            Dictionary with DOM analysis
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract components
        forms = self._extract_forms(soup)
        links = self._extract_links(soup)
        images = self._extract_images(soup)
        scripts = self._extract_scripts(soup)
        
        # Calculate DOM hash
        dom_hash = self._calculate_dom_hash(soup)
        
        # Analyze structure
        structure = self._analyze_structure(soup)
        
        # Extract visible text for NLP analysis (orchestrator passes to NLP service)
        text = self._extract_visible_text(soup)
        
        return {
            "dom_hash": dom_hash,
            "text": text,
            "element_count": len(soup.find_all()),
            "forms": forms,
            "form_count": len(forms),
            "links": links[:50],  # Limit to first 50
            "link_count": len(links),
            "images": images[:20],  # Limit to first 20
            "image_count": len(images),
            "scripts": scripts,
            "script_count": len(scripts),
            "structure": structure,
            "is_suspicious": self._is_suspicious(forms, links, structure)
        }
    
    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract form data"""
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
        """Extract links"""
        links = []
        for link in soup.find_all('a', href=True):
            links.append({
                "href": link['href'],
                "text": link.get_text(strip=True)[:100],  # Limit text length
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
    
    def _extract_scripts(self, soup: BeautifulSoup) -> Dict:
        """Extract script information"""
        scripts = soup.find_all('script')
        external_scripts = [s.get('src') for s in scripts if s.get('src')]
        
        return {
            "total_count": len(scripts),
            "external_count": len(external_scripts),
            "external_scripts": external_scripts[:10]  # First 10
        }
    
    def _calculate_dom_hash(self, soup: BeautifulSoup) -> str:
        """Calculate hash of DOM structure"""
        # Get structure without content
        dom_structure = str([tag.name for tag in soup.find_all()])
        return hashlib.sha256(dom_structure.encode()).hexdigest()[:16]
    
    def _analyze_structure(self, soup: BeautifulSoup) -> Dict:
        """Analyze DOM structure patterns"""
        return {
            "has_login_form": bool(soup.find('form', {'action': lambda x: x and ('login' in x.lower() or 'signin' in x.lower())})),
            "has_password_field": bool(soup.find('input', {'type': 'password'})),
            "has_email_field": bool(soup.find('input', {'type': 'email'})),
            "has_credit_card_field": bool(soup.find('input', {'name': lambda x: x and ('card' in x.lower() or 'cvv' in x.lower())})),
            "iframe_count": len(soup.find_all('iframe')),
            "script_count": len(soup.find_all('script')),
            "external_script_count": len([s for s in soup.find_all('script', src=True) if s.get('src', '').startswith('http')])
        }
    
    def _extract_visible_text(self, soup: BeautifulSoup, max_len: int = 5000) -> str:
        """Extract visible text from page for NLP analysis"""
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text[:max_len] if text else ""

    def _is_suspicious(self, forms: List[Dict], links: List[Dict], structure: Dict) -> bool:
        """Detect suspicious patterns"""
        # Multiple forms
        if len(forms) > 3:
            return True
        
        # Password field present
        if structure.get("has_password_field"):
            return True
        
        # Excessive external links
        external_links = [l for l in links if l.get('is_external', False)]
        if len(external_links) > 20:
            return True
        
        # Many iframes (could be clickjacking)
        if structure.get("iframe_count", 0) > 2:
            return True
        
        return False
