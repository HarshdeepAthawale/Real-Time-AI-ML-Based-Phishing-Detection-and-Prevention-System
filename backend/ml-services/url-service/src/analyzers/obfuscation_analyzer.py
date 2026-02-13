"""
URL obfuscation detection analyzer.
Detects Base64/hex encoding, parameter cloaking, URL shortener chains,
and other obfuscation techniques used in phishing attacks.
"""
import re
import base64
import math
from urllib.parse import urlparse, parse_qs, unquote
from typing import Dict, List


class ObfuscationAnalyzer:
    """Detect URL obfuscation techniques used in phishing."""

    # Known URL shortener domains
    SHORTENER_DOMAINS = {
        "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
        "buff.ly", "adf.ly", "bl.ink", "lnkd.in", "db.tt", "qr.ae",
        "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "clck.ru",
        "v.gd", "tr.im", "x.co", "su.pr", "u.to", "soo.gd",
    }

    # Suspicious TLDs commonly used in phishing
    SUSPICIOUS_TLDS = {
        ".xyz", ".top", ".club", ".online", ".site", ".icu", ".buzz",
        ".tk", ".ml", ".ga", ".cf", ".gq", ".work", ".click", ".link",
        ".info", ".pw", ".cc", ".ws", ".surf", ".monster",
    }

    def __init__(self):
        self._hex_pattern = re.compile(r'%[0-9a-fA-F]{2}')
        self._base64_pattern = re.compile(
            r'[A-Za-z0-9+/]{20,}={0,2}'
        )
        self._ip_pattern = re.compile(
            r'^(?:\d{1,3}\.){3}\d{1,3}$'
        )
        self._long_hex_pattern = re.compile(
            r'(?:0x)?[0-9a-fA-F]{8,}'
        )

    def analyze(self, url: str) -> Dict:
        """
        Analyze URL for obfuscation techniques.

        Returns:
            Dictionary with obfuscation analysis results.
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.split(':')[0]  # strip port
            path = parsed.path
            query = parsed.query
            full_url = url

            results = {
                "is_obfuscated": False,
                "obfuscation_score": 0.0,
                "techniques_detected": [],
                "details": {},
            }

            score = 0.0

            # 1. Base64 detection in path and query
            b64 = self._detect_base64(path, query)
            if b64["detected"]:
                score += 0.25
                results["techniques_detected"].append("base64_encoding")
                results["details"]["base64"] = b64

            # 2. Hex encoding detection
            hex_result = self._detect_hex_encoding(full_url)
            if hex_result["detected"]:
                score += 0.2
                results["techniques_detected"].append("hex_encoding")
                results["details"]["hex_encoding"] = hex_result

            # 3. URL shortener detection
            shortener = self._detect_shortener(domain)
            if shortener["detected"]:
                score += 0.15
                results["techniques_detected"].append("url_shortener")
                results["details"]["url_shortener"] = shortener

            # 4. Parameter cloaking
            cloaking = self._detect_parameter_cloaking(query)
            if cloaking["detected"]:
                score += 0.25
                results["techniques_detected"].append("parameter_cloaking")
                results["details"]["parameter_cloaking"] = cloaking

            # 5. IP address as domain
            ip_result = self._detect_ip_domain(domain)
            if ip_result["detected"]:
                score += 0.2
                results["techniques_detected"].append("ip_address_domain")
                results["details"]["ip_domain"] = ip_result

            # 6. Suspicious TLD
            tld_result = self._detect_suspicious_tld(domain)
            if tld_result["detected"]:
                score += 0.1
                results["techniques_detected"].append("suspicious_tld")
                results["details"]["suspicious_tld"] = tld_result

            # 7. Excessive subdomains
            subdomain_result = self._detect_excessive_subdomains(domain)
            if subdomain_result["detected"]:
                score += 0.15
                results["techniques_detected"].append("excessive_subdomains")
                results["details"]["excessive_subdomains"] = subdomain_result

            # 8. Data URI scheme
            if url.startswith("data:"):
                score += 0.3
                results["techniques_detected"].append("data_uri")

            # 9. @ symbol in URL (credential-style obfuscation)
            if "@" in parsed.netloc:
                score += 0.25
                results["techniques_detected"].append("credential_in_url")
                results["details"]["credential_url"] = {
                    "detected": True,
                    "description": "URL contains @ symbol which can hide the real destination",
                }

            # 10. Double encoding detection
            double_enc = self._detect_double_encoding(full_url)
            if double_enc["detected"]:
                score += 0.2
                results["techniques_detected"].append("double_encoding")
                results["details"]["double_encoding"] = double_enc

            # 11. Punycode / IDN detection
            if domain.startswith("xn--"):
                score += 0.2
                results["techniques_detected"].append("punycode_idn")
                results["details"]["punycode"] = {
                    "detected": True,
                    "domain": domain,
                }

            # Normalize score to 0-1
            results["obfuscation_score"] = min(1.0, score)
            results["is_obfuscated"] = score >= 0.3

            return results

        except Exception as e:
            return {
                "is_obfuscated": False,
                "obfuscation_score": 0.0,
                "techniques_detected": [],
                "details": {},
                "error": str(e),
            }

    def _detect_base64(self, path: str, query: str) -> Dict:
        """Detect Base64-encoded segments in URL path and query."""
        findings = []

        for segment in [path, query]:
            if not segment:
                continue
            matches = self._base64_pattern.findall(segment)
            for match in matches:
                try:
                    decoded = base64.b64decode(match + "==").decode("utf-8", errors="ignore")
                    # Check if decoded content looks like a URL or contains suspicious content
                    if any(c.isalpha() for c in decoded) and len(decoded) > 5:
                        findings.append({
                            "encoded": match[:50],
                            "decoded_preview": decoded[:100],
                        })
                except Exception:
                    pass

        return {
            "detected": len(findings) > 0,
            "count": len(findings),
            "findings": findings[:5],
        }

    def _detect_hex_encoding(self, url: str) -> Dict:
        """Detect excessive hex encoding in URL."""
        hex_matches = self._hex_pattern.findall(url)
        # Decode and check if it's hiding readable text
        decoded = unquote(url)
        encoding_ratio = len(hex_matches) * 3 / max(len(url), 1)

        # Also check for long hex strings (potential encoded payloads)
        long_hex = self._long_hex_pattern.findall(url)

        return {
            "detected": encoding_ratio > 0.15 or len(long_hex) > 0,
            "hex_encoded_chars": len(hex_matches),
            "encoding_ratio": round(encoding_ratio, 3),
            "long_hex_strings": len(long_hex),
            "decoded_differs": decoded != url,
        }

    def _detect_shortener(self, domain: str) -> Dict:
        """Detect URL shortener domains."""
        domain_lower = domain.lower()
        is_shortener = domain_lower in self.SHORTENER_DOMAINS
        return {
            "detected": is_shortener,
            "domain": domain,
        }

    def _detect_parameter_cloaking(self, query: str) -> Dict:
        """Detect suspicious parameter patterns used for cloaking."""
        if not query:
            return {"detected": False}

        params = parse_qs(query)
        suspicious_indicators = []

        for key, values in params.items():
            for value in values:
                # Long encoded values
                if len(value) > 100:
                    suspicious_indicators.append(f"long_value:{key}")

                # URL as parameter value
                if re.match(r'https?://', unquote(value)):
                    suspicious_indicators.append(f"url_in_param:{key}")

                # Base64-looking values
                if self._base64_pattern.match(value):
                    suspicious_indicators.append(f"base64_param:{key}")

                # Email address in parameter
                if re.match(r'[^@]+@[^@]+\.[^@]+', unquote(value)):
                    suspicious_indicators.append(f"email_in_param:{key}")

        # Suspicious parameter names
        suspicious_param_names = {"redirect", "url", "next", "return", "goto", "redir",
                                  "destination", "forward", "target", "continue", "link"}
        for key in params:
            if key.lower() in suspicious_param_names:
                suspicious_indicators.append(f"redirect_param:{key}")

        # Too many parameters
        if len(params) > 10:
            suspicious_indicators.append("excessive_params")

        return {
            "detected": len(suspicious_indicators) > 0,
            "indicators": suspicious_indicators[:10],
            "param_count": len(params),
        }

    def _detect_ip_domain(self, domain: str) -> Dict:
        """Detect if domain is an IP address."""
        is_ip = bool(self._ip_pattern.match(domain))

        # Also check for decimal/octal/hex IP representations
        is_numeric = domain.replace(".", "").isdigit() if domain else False

        return {
            "detected": is_ip or is_numeric,
            "domain": domain,
            "is_direct_ip": is_ip,
        }

    def _detect_suspicious_tld(self, domain: str) -> Dict:
        """Detect suspicious top-level domains."""
        domain_lower = domain.lower()
        for tld in self.SUSPICIOUS_TLDS:
            if domain_lower.endswith(tld):
                return {"detected": True, "tld": tld}
        return {"detected": False}

    def _detect_excessive_subdomains(self, domain: str) -> Dict:
        """Detect excessive subdomain depth."""
        parts = domain.split(".")
        subdomain_count = max(0, len(parts) - 2)
        return {
            "detected": subdomain_count >= 3,
            "subdomain_count": subdomain_count,
            "parts": parts,
        }

    def _detect_double_encoding(self, url: str) -> Dict:
        """Detect double URL encoding (e.g., %2520 = double-encoded space)."""
        double_encoded = re.findall(r'%25[0-9a-fA-F]{2}', url)
        return {
            "detected": len(double_encoded) > 0,
            "count": len(double_encoded),
        }
