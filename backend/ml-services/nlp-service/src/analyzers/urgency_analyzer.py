import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class UrgencyAnalyzer:
    def __init__(self):
        # Urgency keywords and patterns
        self.urgency_keywords = [
            'urgent', 'immediately', 'asap', 'hurry', 'expired', 'expiring',
            'suspended', 'locked', 'verify', 'confirm', 'update', 'action required',
            'limited time', 'act now', 'click here', 'verify account',
            'time sensitive', 'deadline', 'expires today', 'expires tomorrow',
            'your account will be', 'within 24 hours', 'within 48 hours'
        ]
        
        self.time_patterns = [
            r'\d+\s*(hour|day|minute)s?\s*(ago|left|remaining)',
            r'expir(es|ed|ing)',
            r'deadline',
            r'within\s+\d+\s*(hour|day|minute)s?',
            r'expires?\s+(today|tomorrow|soon)',
            r'act\s+(now|immediately|urgently)',
            r'time.?sensitive',
            r'limited\s+time'
        ]
        
        self.threat_keywords = [
            'suspended', 'locked', 'closed', 'terminated', 'deleted',
            'permanently', 'irreversible', 'cannot be undone'
        ]
    
    def analyze(self, text: str) -> Dict:
        """Analyze urgency indicators"""
        text_lower = text.lower()
        
        # Count urgency keywords
        keyword_count = sum(1 for keyword in self.urgency_keywords if keyword in text_lower)
        
        # Check time patterns
        time_pattern_matches = sum(1 for pattern in self.time_patterns if re.search(pattern, text_lower))
        
        # Check threat keywords
        threat_count = sum(1 for keyword in self.threat_keywords if keyword in text_lower)
        
        # Calculate urgency score (0-100)
        urgency_score = min(100, (keyword_count * 10) + (time_pattern_matches * 15) + (threat_count * 12))
        
        # Check for excessive punctuation (urgency indicator)
        exclamation_count = text.count('!')
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Check for repeated urgency words
        repeated_urgency = len(re.findall(r'\b(urgent|immediately|asap|hurry)\b', text_lower, re.IGNORECASE))
        
        # Calculate final urgency score
        if exclamation_count > 3:
            urgency_score += 10
        if caps_words > 5:
            urgency_score += 10
        if repeated_urgency > 1:
            urgency_score += 15
        
        urgency_score = min(100, urgency_score)
        
        return {
            "urgency_score": urgency_score,
            "keyword_count": keyword_count,
            "time_pattern_matches": time_pattern_matches,
            "threat_count": threat_count,
            "exclamation_count": exclamation_count,
            "caps_words_count": caps_words,
            "repeated_urgency_count": repeated_urgency,
            "is_urgent": urgency_score > 50,
            "urgency_level": self._get_urgency_level(urgency_score)
        }
    
    def _get_urgency_level(self, score: float) -> str:
        """Get urgency level as string"""
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        elif score >= 20:
            return "low"
        else:
            return "none"
    
    def extract_time_references(self, text: str) -> List[Dict]:
        """Extract time references from text"""
        time_references = []
        text_lower = text.lower()
        
        # Pattern for time references
        patterns = [
            (r'within\s+(\d+)\s*(hour|day|minute)s?', 'within'),
            (r'in\s+(\d+)\s*(hour|day|minute)s?', 'in'),
            (r'(\d+)\s*(hour|day|minute)s?\s+(ago|left|remaining)', 'duration'),
            (r'expires?\s+(today|tomorrow|soon)', 'expiration')
        ]
        
        for pattern, ref_type in patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                time_references.append({
                    "type": ref_type,
                    "match": match.group(0),
                    "position": match.start()
                })
        
        return time_references
