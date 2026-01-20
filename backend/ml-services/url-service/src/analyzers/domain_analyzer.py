from typing import Dict, List
import re
from datetime import datetime

class DomainAnalyzer:
    def analyze(self, domain: str) -> Dict:
        """Analyze domain characteristics"""
        return {
            "domain": domain,
            "length": len(domain),
            "subdomain_count": domain.count('.'),
            "has_numbers": bool(re.search(r'\d', domain)),
            "has_hyphens": '-' in domain,
            "character_entropy": self._calculate_entropy(domain),
            "suspicious_patterns": self._detect_suspicious_patterns(domain),
            "is_ip_address": self._is_ip_address(domain)
        }
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy"""
        import math
        if not text:
            return 0
        entropy = 0
        for char in set(text):
            p = text.count(char) / len(text)
            entropy -= p * math.log2(p)
        return entropy
    
    def _detect_suspicious_patterns(self, domain: str) -> List[str]:
        """Detect suspicious domain patterns"""
        patterns = []
        
        # Check for homoglyphs (common lookalike characters)
        homoglyph_patterns = [
            (r'[0-9]', 'numbers_in_domain'),
            (r'[а-я]', 'cyrillic_characters'),
            (r'[α-ω]', 'greek_characters'),
        ]
        
        for pattern, name in homoglyph_patterns:
            if re.search(pattern, domain):
                patterns.append(name)
        
        # Check for suspicious keywords
        suspicious_keywords = ['secure', 'verify', 'update', 'account', 'login', 'bank']
        for keyword in suspicious_keywords:
            if keyword in domain.lower():
                patterns.append(f'suspicious_keyword_{keyword}')
        
        return patterns
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return bool(re.match(ip_pattern, domain))
