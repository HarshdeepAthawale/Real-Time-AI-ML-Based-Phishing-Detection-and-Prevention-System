from typing import Dict, List

class FormAnalyzer:
    def analyze(self, forms: List[Dict]) -> Dict:
        """Analyze forms for credential harvesting"""
        results = {
            "form_count": len(forms),
            "credential_harvesting_score": 0.0,
            "suspicious_patterns": [],
            "is_suspicious": False
        }
        
        for form in forms:
            # Check for password fields
            has_password = any(f['type'] == 'password' for f in form.get('fields', []))
            has_email = any(f['type'] == 'email' for f in form.get('fields', []))
            
            if has_password:
                results["credential_harvesting_score"] += 30
            
            if has_email:
                results["credential_harvesting_score"] += 20
            
            # Check for suspicious field names
            suspicious_keywords = ['ssn', 'social', 'credit', 'card', 'cvv', 'pin', 'account']
            for field in form.get('fields', []):
                field_name = field.get('name', '').lower()
                if any(keyword in field_name for keyword in suspicious_keywords):
                    results["credential_harvesting_score"] += 25
                    results["suspicious_patterns"].append(f"suspicious_field_{field_name}")
            
            # Check form action (suspicious if pointing to external domain)
            action = form.get('action', '')
            if action.startswith('http') and not action.startswith('https'):
                results["suspicious_patterns"].append("http_form_action")
                results["credential_harvesting_score"] += 15
        
        results["credential_harvesting_score"] = min(100, results["credential_harvesting_score"])
        results["is_suspicious"] = results["credential_harvesting_score"] > 50
        
        return results
