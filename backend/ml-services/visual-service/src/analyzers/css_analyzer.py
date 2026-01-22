"""CSS pattern analysis"""
from bs4 import BeautifulSoup
from typing import Dict, List


class CSSAnalyzer:
    """Analyze CSS patterns for suspicious styling"""
    
    def analyze(self, html_content: str) -> Dict:
        """
        Analyze CSS patterns
        
        Args:
            html_content: HTML content
            
        Returns:
            Dictionary with CSS analysis
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract style tags
        style_tags = soup.find_all('style')
        inline_styles = soup.find_all(style=True)
        
        # Check for suspicious CSS patterns
        suspicious_patterns = self._detect_suspicious_patterns(style_tags, inline_styles)
        
        return {
            "style_tag_count": len(style_tags),
            "inline_style_count": len(inline_styles),
            "suspicious_patterns": suspicious_patterns,
            "has_hidden_elements": "hidden_elements" in suspicious_patterns,
            "has_overlay": "overlay_detected" in suspicious_patterns
        }
    
    def _detect_suspicious_patterns(self, style_tags: List, inline_styles: List) -> List[str]:
        """Detect suspicious CSS patterns"""
        patterns = []
        
        # Combine all CSS content
        all_css = ""
        for style in style_tags:
            all_css += style.get_text()
        
        css_lower = all_css.lower()
        
        # Check for hidden elements
        if 'display:none' in css_lower or 'visibility:hidden' in css_lower:
            patterns.append("hidden_elements")
        
        # Check for overlays (potential clickjacking)
        if 'position:fixed' in css_lower or 'position:absolute' in css_lower:
            if 'z-index' in css_lower:
                patterns.append("overlay_detected")
        
        # Check for transparency tricks
        if 'opacity:0' in css_lower or 'opacity: 0' in css_lower:
            patterns.append("transparent_elements")
        
        return patterns
