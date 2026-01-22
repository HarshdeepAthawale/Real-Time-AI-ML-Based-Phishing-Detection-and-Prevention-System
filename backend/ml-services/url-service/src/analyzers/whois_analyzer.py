"""WHOIS record analysis"""
import whois
from typing import Dict
from datetime import datetime
from src.utils.logger import logger


class WHOISAnalyzer:
    """Analyze WHOIS data"""
    
    def analyze(self, domain: str) -> Dict:
        """
        Analyze WHOIS data
        
        Args:
            domain: Domain to analyze
            
        Returns:
            Dictionary with WHOIS analysis
        """
        results = {
            "domain": domain,
            "registered": False,
            "registration_date": None,
            "expiration_date": None,
            "age_days": None,
            "registrar": None,
            "is_suspicious": False
        }
        
        try:
            w = whois.whois(domain)
            
            if w.domain_name:
                results["registered"] = True
                results["registrar"] = w.registrar
                
                # Registration date
                if w.creation_date:
                    reg_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
                    if isinstance(reg_date, datetime):
                        results["registration_date"] = reg_date.isoformat()
                        age = (datetime.now() - reg_date).days
                        results["age_days"] = age
                        
                        # Suspicious if domain is very new (< 30 days)
                        if age < 30:
                            results["is_suspicious"] = True
                
                # Expiration date
                if w.expiration_date:
                    exp_date = w.expiration_date[0] if isinstance(w.expiration_date, list) else w.expiration_date
                    if isinstance(exp_date, datetime):
                        results["expiration_date"] = exp_date.isoformat()
        
        except Exception as e:
            logger.debug(f"WHOIS lookup failed for {domain}: {e}")
            results["error"] = "WHOIS lookup failed"
        
        return results
