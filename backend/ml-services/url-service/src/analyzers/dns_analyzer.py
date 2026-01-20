import dns.resolver
import dns.reversename
from typing import Dict, List, Optional
from datetime import datetime

class DNSAnalyzer:
    def analyze(self, domain: str) -> Dict:
        """Analyze DNS records for domain"""
        results = {
            "domain": domain,
            "a_records": [],
            "aaaa_records": [],
            "mx_records": [],
            "txt_records": [],
            "ns_records": [],
            "cname_records": [],
            "dns_age_days": None,
            "has_dns": False
        }
        
        try:
            # A records
            try:
                a_records = dns.resolver.resolve(domain, 'A')
                results["a_records"] = [str(rdata) for rdata in a_records]
                results["has_dns"] = True
            except:
                pass
            
            # AAAA records (IPv6)
            try:
                aaaa_records = dns.resolver.resolve(domain, 'AAAA')
                results["aaaa_records"] = [str(rdata) for rdata in aaaa_records]
            except:
                pass
            
            # MX records
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                results["mx_records"] = [
                    {"priority": rdata.preference, "exchange": str(rdata.exchange)}
                    for rdata in mx_records
                ]
            except:
                pass
            
            # TXT records (for SPF, DKIM, etc.)
            try:
                txt_records = dns.resolver.resolve(domain, 'TXT')
                results["txt_records"] = [str(rdata) for rdata in txt_records]
            except:
                pass
            
            # NS records
            try:
                ns_records = dns.resolver.resolve(domain, 'NS')
                results["ns_records"] = [str(rdata) for rdata in ns_records]
            except:
                pass
            
            # CNAME records
            try:
                cname_records = dns.resolver.resolve(domain, 'CNAME')
                results["cname_records"] = [str(rdata) for rdata in cname_records]
            except:
                pass
            
            # Calculate DNS age (first seen)
            # This would require historical DNS data
            # For now, mark as unknown
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
