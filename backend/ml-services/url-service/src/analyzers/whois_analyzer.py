import whois
from typing import Dict, Optional
from datetime import datetime

class WHOISAnalyzer:
    def analyze(self, domain: str) -> Dict:
        """Analyze WHOIS data"""
        results = {
            "domain": domain,
            "registered": False,
            "registration_date": None,
            "expiration_date": None,
            "age_days": None,
            "registrar": None,
            "registrant": None,
            "is_suspicious": False
        }
        
        try:
            w = whois.whois(domain)
            
            if w.domain_name:
                results["registered"] = True
                results["registrar"] = w.registrar
                
                # Registration date
                if w.creation_date:
                    if isinstance(w.creation_date, list):
                        reg_date = w.creation_date[0]
                    else:
                        reg_date = w.creation_date
                    results["registration_date"] = reg_date.isoformat() if hasattr(reg_date, 'isoformat') else str(reg_date)
                    
                    # Calculate age
                    if isinstance(reg_date, datetime):
                        age = (datetime.now() - reg_date).days
                        results["age_days"] = age
                        # Suspicious if domain is very new (< 30 days)
                        if age < 30:
                            results["is_suspicious"] = True
                
                # Expiration date
                if w.expiration_date:
                    if isinstance(w.expiration_date, list):
                        exp_date = w.expiration_date[0]
                    else:
                        exp_date = w.expiration_date
                    results["expiration_date"] = exp_date.isoformat() if hasattr(exp_date, 'isoformat') else str(exp_date)
                
                # Check for privacy protection (suspicious)
                if w.registrant and 'privacy' in str(w.registrant).lower():
                    results["is_suspicious"] = True
                
        except Exception as e:
            results["error"] = str(e)
        
        return results
