"""Homoglyph attack detection"""
import unicodedata
from typing import Dict, List
from difflib import SequenceMatcher


class HomoglyphDetector:
    """Detect homoglyph attacks (lookalike characters)"""
    
    def __init__(self):
        # Common homoglyph mappings
        self.homoglyphs = {
            'o': ['0', 'о', 'ο', 'ο'],
            'i': ['1', 'l', 'і', 'ι', 'ı'],
            'a': ['а', 'α', '@'],
            'e': ['е', 'е', '3'],
            'p': ['р', 'ρ'],
            'c': ['с', 'ς'],
            'x': ['х', 'χ'],
            'y': ['у', 'γ'],
            's': ['$', '5'],
            'g': ['9', 'q']
        }
    
    def detect(self, domain: str, legitimate_domain: str = None) -> Dict:
        """
        Detect homoglyph attacks
        
        Args:
            domain: Domain to check
            legitimate_domain: Known legitimate domain to compare against
            
        Returns:
            Dictionary with detection results
        """
        results = {
            "has_homoglyphs": False,
            "homoglyph_chars": [],
            "similarity_score": 0.0,
            "is_suspicious": False
        }
        
        # Check for non-ASCII characters
        homoglyph_chars = []
        for char in domain:
            if ord(char) > 127:  # Non-ASCII
                homoglyph_chars.append({
                    "char": char,
                    "unicode_name": unicodedata.name(char, "UNKNOWN"),
                    "category": unicodedata.category(char)
                })
        
        if homoglyph_chars:
            results["has_homoglyphs"] = True
            results["homoglyph_chars"] = homoglyph_chars
            results["is_suspicious"] = True
        
        # Compare with legitimate domain if provided
        if legitimate_domain:
            similarity = self._calculate_similarity(domain, legitimate_domain)
            results["similarity_score"] = similarity
            
            # High similarity but not identical is very suspicious
            if similarity > 0.8 and domain != legitimate_domain:
                results["is_suspicious"] = True
                results["likely_spoofing"] = legitimate_domain
        
        return results
    
    def _calculate_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate visual similarity between domains"""
        return SequenceMatcher(None, domain1.lower(), domain2.lower()).ratio()
    
    def check_against_popular_domains(self, domain: str) -> Dict:
        """Check if domain is similar to popular brands"""
        popular_domains = [
            'google.com', 'paypal.com', 'microsoft.com', 'apple.com',
            'amazon.com', 'facebook.com', 'netflix.com', 'linkedin.com',
            'instagram.com', 'twitter.com', 'github.com', 'dropbox.com'
        ]
        
        matches = []
        for popular in popular_domains:
            similarity = self._calculate_similarity(domain, popular)
            if similarity > 0.7 and domain != popular:
                matches.append({
                    "domain": popular,
                    "similarity": similarity
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            "has_matches": len(matches) > 0,
            "matches": matches[:5],  # Top 5
            "is_brand_spoofing": len(matches) > 0
        }
