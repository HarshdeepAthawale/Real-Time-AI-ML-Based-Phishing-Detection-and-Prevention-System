"""Form field analysis for credential harvesting"""
from typing import Dict, List


class FormAnalyzer:
    """Analyze forms for credential harvesting"""
    
    def analyze(self, forms: List[Dict]) -> Dict:
        """
        Analyze forms for suspicious patterns
        
        Args:
            forms: List of form dictionaries from DOM analyzer
            
        Returns:
            Dictionary with form analysis
        """
        results = {
            "form_count": len(forms),
            "credential_harvesting_score": 0.0,
            "suspicious_patterns": [],
            "is_suspicious": False,
            "forms_with_password": 0,
            "forms_with_credit_card": 0
        }
        
        for form in forms:
            fields = form.get('fields', [])
            
            # Check for password fields
            has_password = any(f.get('type') == 'password' for f in fields)
            if has_password:
                results["credential_harvesting_score"] += 30
                results["forms_with_password"] += 1
            
            # Check for email fields
            has_email = any(f.get('type') == 'email' for f in fields)
            if has_email:
                results["credential_harvesting_score"] += 20
            
            # Check for suspicious field names
            suspicious_keywords = ['ssn', 'social', 'credit', 'card', 'cvv', 'pin', 'account', 'routing']
            for field in fields:
                field_name = field.get('name', '').lower()
                field_placeholder = field.get('placeholder', '').lower()
                
                if any(keyword in field_name or keyword in field_placeholder for keyword in suspicious_keywords):
                    results["credential_harvesting_score"] += 25
                    results["suspicious_patterns"].append(f"suspicious_field_{field_name}")
                    
                    if 'card' in field_name or 'cvv' in field_name:
                        results["forms_with_credit_card"] += 1
            
            # Check form action (suspicious if HTTP instead of HTTPS)
            action = form.get('action', '')
            if action.startswith('http://'):
                results["suspicious_patterns"].append("http_form_action")
                results["credential_harvesting_score"] += 15
            
            # Check if action points to external domain
            if action.startswith('http') and '://' in action:
                results["suspicious_patterns"].append("external_form_action")
                results["credential_harvesting_score"] += 10
        
        # Cap score at 100
        results["credential_harvesting_score"] = min(100, results["credential_harvesting_score"])
        results["is_suspicious"] = results["credential_harvesting_score"] > 50
        
        return results
