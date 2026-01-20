import ssl
import socket
from typing import Dict, Optional
from datetime import datetime
from cryptography import x509
from cryptography.hazmat.backends import default_backend

class SSLAnalyzer:
    def analyze(self, domain: str, port: int = 443) -> Dict:
        """Analyze SSL certificate for domain"""
        results = {
            "domain": domain,
            "port": port,
            "has_ssl": False,
            "certificate_valid": False,
            "certificate_issuer": None,
            "certificate_subject": None,
            "certificate_expiry": None,
            "days_until_expiry": None,
            "is_suspicious": False
        }
        
        try:
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect to domain
            with socket.create_connection((domain, port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert(binary_form=True)
                    
                    if cert:
                        results["has_ssl"] = True
                        
                        # Parse certificate
                        cert_obj = x509.load_der_x509_certificate(cert, default_backend())
                        
                        # Get issuer
                        issuer = cert_obj.issuer.rfc4514_string()
                        results["certificate_issuer"] = issuer
                        
                        # Get subject
                        subject = cert_obj.subject.rfc4514_string()
                        results["certificate_subject"] = subject
                        
                        # Get expiry date
                        expiry = cert_obj.not_valid_after
                        results["certificate_expiry"] = expiry.isoformat()
                        
                        # Calculate days until expiry
                        days_until_expiry = (expiry - datetime.now()).days
                        results["days_until_expiry"] = days_until_expiry
                        
                        # Check if certificate is valid
                        now = datetime.now()
                        if cert_obj.not_valid_before <= now <= cert_obj.not_valid_after:
                            results["certificate_valid"] = True
                        else:
                            results["is_suspicious"] = True
                        
                        # Suspicious if certificate expires soon (< 30 days)
                        if days_until_expiry < 30:
                            results["is_suspicious"] = True
                        
                        # Check for self-signed certificates
                        if 'self-signed' in issuer.lower() or issuer == subject:
                            results["is_suspicious"] = True
                            
        except socket.timeout:
            results["error"] = "Connection timeout"
        except socket.gaierror:
            results["error"] = "DNS resolution failed"
        except ssl.SSLError as e:
            results["error"] = f"SSL error: {str(e)}"
            results["is_suspicious"] = True
        except Exception as e:
            results["error"] = str(e)
        
        return results
