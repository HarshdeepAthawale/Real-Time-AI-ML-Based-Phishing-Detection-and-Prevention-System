"""SSL certificate analysis"""
import ssl
import socket
from typing import Dict
from datetime import datetime
from src.utils.logger import logger


class SSLAnalyzer:
    """Analyze SSL certificates"""
    
    def analyze(self, domain: str, port: int = 443) -> Dict:
        """
        Analyze SSL certificate
        
        Args:
            domain: Domain to analyze
            port: Port number (default 443)
            
        Returns:
            Dictionary with SSL analysis
        """
        results = {
            "domain": domain,
            "has_ssl": False,
            "is_valid": False,
            "issuer": None,
            "subject": None,
            "expiration_date": None,
            "days_until_expiration": None
        }
        
        try:
            context = ssl.create_default_context()
            
            with socket.create_connection((domain, port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    
                    results["has_ssl"] = True
                    results["is_valid"] = True
                    
                    # Extract issuer
                    if cert.get('issuer'):
                        issuer_dict = dict(x[0] for x in cert['issuer'])
                        results["issuer"] = issuer_dict.get('organizationName', 'Unknown')
                    
                    # Extract subject
                    if cert.get('subject'):
                        subject_dict = dict(x[0] for x in cert['subject'])
                        results["subject"] = subject_dict.get('commonName', domain)
                    
                    # Check expiration
                    if cert.get('notAfter'):
                        exp_date = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        results["expiration_date"] = exp_date.isoformat()
                        days_until_exp = (exp_date - datetime.now()).days
                        results["days_until_expiration"] = days_until_exp
                        
                        # Suspicious if certificate is expiring soon
                        if days_until_exp < 30:
                            results["expiring_soon"] = True
        
        except ssl.SSLError as e:
            logger.debug(f"SSL error for {domain}: {e}")
            results["ssl_error"] = str(e)
        
        except socket.timeout:
            logger.debug(f"SSL connection timeout for {domain}")
            results["error"] = "Connection timeout"
        
        except Exception as e:
            logger.debug(f"SSL analysis failed for {domain}: {e}")
            results["error"] = str(e)
        
        return results
