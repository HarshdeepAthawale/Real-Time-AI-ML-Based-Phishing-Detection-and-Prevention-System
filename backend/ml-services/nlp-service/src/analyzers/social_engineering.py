"""Social engineering indicator detection"""
from typing import Dict, List


class SocialEngineeringAnalyzer:
    """Detect social engineering tactics"""
    
    def __init__(self):
        # Authority impersonation indicators
        self.authority_keywords = [
            'bank', 'paypal', 'microsoft', 'google', 'amazon', 'apple',
            'irs', 'fbi', 'police', 'government', 'tax', 'security team',
            'support team', 'administrator', 'manager', 'ceo', 'president'
        ]
        
        # Action request indicators
        self.action_keywords = [
            'click here', 'verify now', 'confirm now', 'update now',
            'download', 'install', 'login', 'sign in', 'enter your',
            'provide your', 'send us', 'call us', 'reply with'
        ]
        
        # Reward/prize indicators
        self.reward_keywords = [
            'won', 'winner', 'prize', 'reward', 'free', 'gift',
            'congratulations', 'selected', 'claim', 'bonus'
        ]
        
        # Threat indicators
        self.threat_keywords = [
            'suspend', 'close', 'terminate', 'cancel', 'delete',
            'expire', 'lock', 'block', 'restrict', 'penalty'
        ]
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze social engineering indicators
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with social engineering analysis
        """
        text_lower = text.lower()
        
        # Count different tactic types
        authority_count = sum(1 for kw in self.authority_keywords if kw in text_lower)
        action_count = sum(1 for kw in self.action_keywords if kw in text_lower)
        reward_count = sum(1 for kw in self.reward_keywords if kw in text_lower)
        threat_count = sum(1 for kw in self.threat_keywords if kw in text_lower)
        
        # Detect specific tactics
        tactics = []
        
        if authority_count > 0:
            tactics.append("authority_impersonation")
        
        if action_count > 2:
            tactics.append("urgency_action_requests")
        
        if reward_count > 0:
            tactics.append("reward_baiting")
        
        if threat_count > 1:
            tactics.append("threat_tactics")
        
        # Combination tactics are highly suspicious
        if authority_count > 0 and threat_count > 0:
            tactics.append("authority_with_threat")
        
        if reward_count > 0 and action_count > 0:
            tactics.append("reward_with_urgency")
        
        # Calculate overall social engineering score (0-100)
        se_score = min(100, (
            authority_count * 15 +
            action_count * 10 +
            reward_count * 20 +
            threat_count * 15
        ))
        
        return {
            "social_engineering_score": se_score,
            "tactics_detected": tactics,
            "authority_impersonation_count": authority_count,
            "action_request_count": action_count,
            "reward_mention_count": reward_count,
            "threat_count": threat_count,
            "is_social_engineering": se_score > 40,
            "risk_level": "high" if se_score > 60 else ("medium" if se_score > 30 else "low")
        }
