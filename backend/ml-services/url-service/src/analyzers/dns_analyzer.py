"""DNS record analysis"""
import dns.resolver
from typing import Dict
from src.config import settings
from src.utils.logger import logger


class DNSAnalyzer:
    """Analyze DNS records for domain"""
    
    def __init__(self):
        self.timeout = settings.dns_timeout
    
    def analyze(self, domain: str) -> Dict:
        """
        Analyze DNS records
        
        Args:
            domain: Domain to analyze
            
        Returns:
            Dictionary with DNS analysis
        """
        results = {
            "domain": domain,
            "a_records": [],
            "aaaa_records": [],
            "mx_records": [],
            "txt_records": [],
            "ns_records": [],
            "has_dns": False,
            "record_count": 0
        }
        
        try:
            # Configure resolver timeout
            resolver = dns.resolver.Resolver()
            resolver.timeout = self.timeout
            resolver.lifetime = self.timeout
            
            # A records (IPv4)
            try:
                a_records = resolver.resolve(domain, 'A')
                results["a_records"] = [str(rdata) for rdata in a_records]
                results["has_dns"] = True
            except Exception:
                pass
            
            # AAAA records (IPv6)
            try:
                aaaa_records = resolver.resolve(domain, 'AAAA')
                results["aaaa_records"] = [str(rdata) for rdata in aaaa_records]
            except Exception:
                pass
            
            # MX records (Mail servers)
            try:
                mx_records = resolver.resolve(domain, 'MX')
                results["mx_records"] = [
                    {"priority": rdata.preference, "exchange": str(rdata.exchange)}
                    for rdata in mx_records
                ]
            except Exception:
                pass
            
            # TXT records (SPF, DKIM, etc.)
            try:
                txt_records = resolver.resolve(domain, 'TXT')
                results["txt_records"] = [str(rdata) for rdata in txt_records]
            except Exception:
                pass
            
            # NS records (Name servers)
            try:
                ns_records = resolver.resolve(domain, 'NS')
                results["ns_records"] = [str(rdata) for rdata in ns_records]
            except Exception:
                pass
            
            # Count total records
            results["record_count"] = (
                len(results["a_records"]) +
                len(results["aaaa_records"]) +
                len(results["mx_records"]) +
                len(results["txt_records"]) +
                len(results["ns_records"])
            )
        
        except Exception as e:
            logger.error(f"DNS analysis error for {domain}: {e}")
            results["error"] = str(e)
        
        return results
