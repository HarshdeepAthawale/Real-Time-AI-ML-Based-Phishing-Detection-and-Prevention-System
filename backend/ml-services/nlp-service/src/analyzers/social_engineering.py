import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SocialEngineeringAnalyzer:
    """Analyze social engineering indicators in text"""
    
    def __init__(self):
        # Authority impersonation keywords
        self.authority_keywords = [
            'irs', 'fbi', 'police', 'government', 'court', 'legal',
            'law enforcement', 'tax department', 'revenue service',
            'bank security', 'fraud department', 'security team'
        ]
        
        # Trust-building keywords
        self.trust_keywords = [
            'verified', 'secure', 'official', 'legitimate', 'trusted',
            'certified', 'guaranteed', 'safe', 'protected'
        ]
        
        # Fear and urgency tactics
        self.fear_keywords = [
            'arrest', 'lawsuit', 'legal action', 'criminal charges',
            'account closure', 'permanent ban', 'irreversible',
            'cannot be undone', 'immediate consequences'
        ]
        
        # Social proof indicators
        self.social_proof_patterns = [
            r'\d+\s+people?\s+(viewed|bought|downloaded)',
            r'limited\s+to\s+\d+',
            r'only\s+\d+\s+left',
            r'expires?\s+in\s+\d+'
        ]
        
        # Scarcity indicators
        self.scarcity_keywords = [
            'limited time', 'limited offer', 'only today', 'expires soon',
            'while supplies last', 'first come first serve', 'exclusive'
        ]
        
        # Reciprocity indicators
        self.reciprocity_keywords = [
            'free', 'bonus', 'gift', 'reward', 'prize', 'special offer',
            'exclusive deal', 'just for you', 'congratulations'
        ]
    
    def analyze(self, text: str) -> Dict:
        """Analyze social engineering indicators"""
        text_lower = text.lower()
        
        # Authority impersonation
        authority_count = sum(1 for keyword in self.authority_keywords if keyword in text_lower)
        has_authority_impersonation = authority_count > 0
        
        # Trust-building tactics
        trust_count = sum(1 for keyword in self.trust_keywords if keyword in text_lower)
        
        # Fear tactics
        fear_count = sum(1 for keyword in self.fear_keywords if keyword in text_lower)
        has_fear_tactics = fear_count > 0
        
        # Social proof
        social_proof_matches = sum(1 for pattern in self.social_proof_patterns if re.search(pattern, text_lower))
        has_social_proof = social_proof_matches > 0
        
        # Scarcity
        scarcity_count = sum(1 for keyword in self.scarcity_keywords if keyword in text_lower)
        has_scarcity = scarcity_count > 0
        
        # Reciprocity
        reciprocity_count = sum(1 for keyword in self.reciprocity_keywords if keyword in text_lower)
        has_reciprocity = reciprocity_count > 0
        
        # Calculate overall social engineering score (0-100)
        se_score = 0
        se_score += min(30, authority_count * 10)
        se_score += min(20, trust_count * 5)
        se_score += min(25, fear_count * 8)
        se_score += min(15, social_proof_matches * 5)
        se_score += min(10, scarcity_count * 3)
        se_score += min(10, reciprocity_count * 2)
        
        se_score = min(100, se_score)
        
        # Identify tactics used
        tactics = []
        if has_authority_impersonation:
            tactics.append("authority_impersonation")
        if has_fear_tactics:
            tactics.append("fear_appeal")
        if has_social_proof:
            tactics.append("social_proof")
        if has_scarcity:
            tactics.append("scarcity")
        if has_reciprocity:
            tactics.append("reciprocity")
        
        return {
            "social_engineering_score": se_score,
            "authority_impersonation": {
                "detected": has_authority_impersonation,
                "count": authority_count
            },
            "trust_building": {
                "detected": trust_count > 0,
                "count": trust_count
            },
            "fear_appeal": {
                "detected": has_fear_tactics,
                "count": fear_count
            },
            "social_proof": {
                "detected": has_social_proof,
                "matches": social_proof_matches
            },
            "scarcity": {
                "detected": has_scarcity,
                "count": scarcity_count
            },
            "reciprocity": {
                "detected": has_reciprocity,
                "count": reciprocity_count
            },
            "tactics_used": tactics,
            "is_social_engineering": se_score > 50
        }
    
    def extract_authority_claims(self, text: str) -> List[Dict]:
        """Extract authority claims from text"""
        text_lower = text.lower()
        claims = []
        
        for keyword in self.authority_keywords:
            if keyword in text_lower:
                # Find context around the keyword
                pattern = rf'.{{0,50}}{re.escape(keyword)}.{{0,50}}'
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    claims.append({
                        "authority": keyword,
                        "context": match.group(0).strip()
                    })
        
        return claims
    
    def detect_impersonation(self, text: str, sender_info: Dict = None) -> Dict:
        """Detect impersonation attempts"""
        text_lower = text.lower()
        
        # Check for mismatched authority claims
        has_authority_claim = any(keyword in text_lower for keyword in self.authority_keywords)
        
        # Check sender domain if provided
        sender_domain_mismatch = False
        if sender_info and 'domain' in sender_info:
            sender_domain = sender_info['domain'].lower()
            # Common legitimate domains
            legitimate_domains = ['gov', 'edu', 'org', 'com']
            if has_authority_claim and not any(legit in sender_domain for legit in legitimate_domains):
                sender_domain_mismatch = True
        
        return {
            "has_authority_claim": has_authority_claim,
            "sender_domain_mismatch": sender_domain_mismatch,
            "likely_impersonation": has_authority_claim and sender_domain_mismatch
        }
