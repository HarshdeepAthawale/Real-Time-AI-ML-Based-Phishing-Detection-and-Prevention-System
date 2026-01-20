from typing import Dict, List, Tuple
import unicodedata

class HomoglyphDetector:
    def __init__(self):
        # Common homoglyph mappings
        self.homoglyphs = {
            'o': ['0', 'о', 'ο', 'о'],
            'i': ['1', 'l', 'і', 'ι', 'ı'],
            'a': ['а', 'α'],
            'e': ['е', 'е'],
            'p': ['р', 'ρ'],
            'c': ['с', 'ς'],
            'x': ['х', 'χ'],
            'y': ['у', 'γ'],
        }
    
    def detect(self, domain: str, legitimate_domain: str = None) -> Dict:
        """Detect homoglyph attacks"""
        results = {
            "has_homoglyphs": False,
            "homoglyph_chars": [],
            "similarity_score": 0.0,
            "is_suspicious": False
        }
        
        if legitimate_domain:
            similarity = self._calculate_similarity(domain, legitimate_domain)
            results["similarity_score"] = similarity
            results["is_suspicious"] = similarity > 0.8 and domain != legitimate_domain
        
        # Check for homoglyph characters
        homoglyph_chars = []
        for char in domain:
            normalized = unicodedata.normalize('NFKD', char)
            if normalized != char:
                homoglyph_chars.append({
                    "char": char,
                    "normalized": normalized,
                    "unicode_name": unicodedata.name(char, "UNKNOWN")
                })
        
        if homoglyph_chars:
            results["has_homoglyphs"] = True
            results["homoglyph_chars"] = homoglyph_chars
            results["is_suspicious"] = True
        
        return results
    
    def _calculate_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate visual similarity between domains"""
        # Simple Levenshtein distance normalized
        from difflib import SequenceMatcher
        return SequenceMatcher(None, domain1.lower(), domain2.lower()).ratio()
