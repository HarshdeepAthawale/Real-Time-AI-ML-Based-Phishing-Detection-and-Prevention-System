from bs4 import BeautifulSoup
from typing import Dict, List
import re

class CSSAnalyzer:
    def analyze(self, html_content: str) -> Dict:
        """Analyze CSS patterns for suspicious indicators"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract inline styles
        inline_styles = self._extract_inline_styles(soup)
        
        # Extract style tags
        style_tags = self._extract_style_tags(soup)
        
        # Extract external stylesheets
        external_stylesheets = self._extract_external_stylesheets(soup)
        
        # Analyze suspicious patterns
        suspicious_patterns = self._detect_suspicious_patterns(inline_styles, style_tags)
        
        return {
            "inline_style_count": len(inline_styles),
            "style_tag_count": len(style_tags),
            "external_stylesheet_count": len(external_stylesheets),
            "suspicious_patterns": suspicious_patterns,
            "is_suspicious": len(suspicious_patterns) > 0
        }
    
    def _extract_inline_styles(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract inline style attributes"""
        inline_styles = []
        for element in soup.find_all(style=True):
            inline_styles.append({
                "tag": element.name,
                "style": element.get('style', ''),
                "id": element.get('id', ''),
                "class": element.get('class', [])
            })
        return inline_styles
    
    def _extract_style_tags(self, soup: BeautifulSoup) -> List[str]:
        """Extract content from <style> tags"""
        style_tags = []
        for style_tag in soup.find_all('style'):
            style_tags.append(style_tag.string or '')
        return style_tags
    
    def _extract_external_stylesheets(self, soup: BeautifulSoup) -> List[str]:
        """Extract external stylesheet links"""
        stylesheets = []
        for link in soup.find_all('link', rel='stylesheet'):
            href = link.get('href', '')
            if href:
                stylesheets.append(href)
        return stylesheets
    
    def _detect_suspicious_patterns(self, inline_styles: List[Dict], style_tags: List[str]) -> List[str]:
        """Detect suspicious CSS patterns"""
        suspicious = []
        
        # Check for display:none (common in phishing to hide elements)
        display_none_pattern = re.compile(r'display\s*:\s*none', re.IGNORECASE)
        
        for style_item in inline_styles + [{"style": tag} for tag in style_tags]:
            style_content = style_item.get('style', '')
            
            if display_none_pattern.search(style_content):
                suspicious.append("display_none_detected")
            
            # Check for very small font sizes (obfuscation)
            if re.search(r'font-size\s*:\s*0(px|em|rem)', style_content, re.IGNORECASE):
                suspicious.append("zero_font_size")
            
            # Check for opacity:0 (hiding content)
            if re.search(r'opacity\s*:\s*0', style_content, re.IGNORECASE):
                suspicious.append("zero_opacity")
            
            # Check for position:absolute with off-screen coordinates
            if re.search(r'position\s*:\s*absolute', style_content, re.IGNORECASE):
                if re.search(r'(left|top)\s*:\s*-?\d+px', style_content, re.IGNORECASE):
                    suspicious.append("off_screen_positioning")
        
        return list(set(suspicious))  # Remove duplicates
