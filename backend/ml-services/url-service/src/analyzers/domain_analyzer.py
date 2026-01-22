"""Domain characteristics analyzer"""
from typing import Dict, List
import re
import math


class DomainAnalyzer:
    """Analyze domain characteristics"""
    
    def analyze(self, domain: str) -> Dict:
        """
        Analyze domain characteristics
        
        Args:
            domain: Domain name to analyze
            
        Returns:
            Dictionary with domain analysis
        """
        return {
            "domain": domain,
            "length": len(domain),
            "subdomain_count": domain.count('.'),
            "has_numbers": bool(re.search(r'\d', domain)),
            "has_hyphens": '-' in domain,
            "character_entropy": self._calculate_entropy(domain),
            "suspicious_patterns": self._detect_suspicious_patterns(domain),
            "is_ip_address": self._is_ip_address(domain),
            "vowel_consonant_ratio": self._vowel_consonant_ratio(domain)
        }
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy"""
        if not text:
            return 0
        
        entropy = 0
        for char in set(text):
            p = text.count(char) / len(text)
            entropy -= p * math.log2(p)
        
        return round(entropy, 2)
    
    def _detect_suspicious_patterns(self, domain: str) -> List[str]:
        """Detect suspicious domain patterns"""
        patterns = []
        
        # Check for numbers
        if re.search(r'\d', domain):
            patterns.append('contains_numbers')
        
        # Check for excessive hyphens
        if domain.count('-') > 2:
            patterns.append('excessive_hyphens')
        
        # Check for suspicious keywords
        suspicious_keywords = ['secure', 'verify', 'update', 'account', 'login', 'bank', 'paypal', 'microsoft']
        for keyword in suspicious_keywords:
            if keyword in domain.lower():
                patterns.append(f'suspicious_keyword_{keyword}')
        
        # Check for homoglyph characters (basic detection)
        if any(ord(c) > 127 for c in domain):
            patterns.append('non_ascii_characters')
        
        # Check for excessive length
        if len(domain) > 30:
            patterns.append('excessive_length')
        
        return patterns
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        return bool(re.match(ip_pattern, domain))
    
    def _vowel_consonant_ratio(self, domain: str) -> float:
        """Calculate vowel to consonant ratio"""
        vowels = 'aeiou'
        vowel_count = sum(1 for c in domain.lower() if c in vowels)
        consonant_count = sum(1 for c in domain.lower() if c.isalpha() and c not in vowels)
        
        if consonant_count == 0:
            return 0.0
        
        return round(vowel_count / consonant_count, 2)
