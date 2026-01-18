"""
URL/Domain Analysis Service - Phishing Detection using GNN models
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import os
import re
import time
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="URL Phishing Detection Service", version="1.0.0")


class URLAnalysisRequest(BaseModel):
    url: str
    domain: Optional[str] = None


class URLAnalysisResponse(BaseModel):
    is_phishing: bool
    confidence: float
    features: Dict
    model_version: str
    domain_info: Optional[Dict] = None
    processing_time_ms: Optional[float] = None


# Suspicious TLDs
SUSPICIOUS_TLDS = [".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click"]

# Suspicious domain patterns
SUSPICIOUS_PATTERNS = [
    r"bit\.ly", r"tinyurl", r"t\.co", r"goo\.gl", r"short\.link",
    r"\d+\.\d+\.\d+\.\d+",  # IP addresses
    r"[a-z0-9-]+\.(tk|ml|ga|cf|gq)",  # Suspicious TLDs
]


def analyze_domain(domain: str) -> Dict:
    """Analyze domain for suspicious characteristics"""
    domain_lower = domain.lower()
    
    # Check for suspicious TLDs
    has_suspicious_tld = any(tld in domain_lower for tld in SUSPICIOUS_TLDS)
    
    # Check for IP address
    is_ip = bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', domain))
    
    # Check domain length (very long domains are suspicious)
    is_long_domain = len(domain) > 50
    
    # Check for subdomain count (many subdomains are suspicious)
    subdomain_count = domain.count('.')
    has_many_subdomains = subdomain_count > 3
    
    # Check for homoglyphs or suspicious characters
    has_suspicious_chars = bool(re.search(r'[0-9]+[a-z]+|[a-z]+[0-9]+', domain))
    
    return {
        "has_suspicious_tld": has_suspicious_tld,
        "is_ip": is_ip,
        "is_long_domain": is_long_domain,
        "has_many_subdomains": has_many_subdomains,
        "has_suspicious_chars": has_suspicious_chars,
        "subdomain_count": subdomain_count,
        "domain_length": len(domain)
    }


def extract_url_features(url: str) -> Dict:
    """Extract features from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        
        # Analyze domain
        domain_info = analyze_domain(domain)
        
        # URL length
        url_length = len(url)
        
        # Check for suspicious patterns
        has_suspicious_pattern = any(re.search(pattern, url, re.IGNORECASE) for pattern in SUSPICIOUS_PATTERNS)
        
        # Check for URL shortening services
        is_shortened = any(service in url.lower() for service in ["bit.ly", "tinyurl", "t.co", "goo.gl", "short.link"])
        
        # Check for HTTPS
        has_https = url.startswith("https://")
        
        # Check for port number
        has_port = ":" in parsed.netloc and not parsed.netloc.endswith(":443") and not parsed.netloc.endswith(":80")
        
        # Check query parameters
        query_params = parse_qs(parsed.query)
        has_query_params = len(query_params) > 0
        
        return {
            "url_length": url_length,
            "has_suspicious_pattern": has_suspicious_pattern,
            "is_shortened": is_shortened,
            "has_https": has_https,
            "has_port": has_port,
            "has_query_params": has_query_params,
            "domain_info": domain_info
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_phishing_score(features: Dict) -> float:
    """Calculate phishing probability based on URL features"""
    score = 0.0
    
    domain_info = features.get("domain_info", {})
    
    # Suspicious TLD
    if domain_info.get("has_suspicious_tld", False):
        score += 0.3
    
    # IP address instead of domain
    if domain_info.get("is_ip", False):
        score += 0.25
    
    # Very long domain
    if domain_info.get("is_long_domain", False):
        score += 0.15
    
    # Many subdomains
    if domain_info.get("has_many_subdomains", False):
        score += 0.1
    
    # Suspicious pattern match
    if features.get("has_suspicious_pattern", False):
        score += 0.2
    
    # URL shortening service
    if features.get("is_shortened", False):
        score += 0.15
    
    # No HTTPS
    if not features.get("has_https", False):
        score += 0.1
    
    # Non-standard port
    if features.get("has_port", False):
        score += 0.1
    
    return min(score, 1.0)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "url-service"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "url-service", "version": "1.0.0"}


@app.post("/analyze", response_model=URLAnalysisResponse)
async def analyze_url(request: URLAnalysisRequest):
    """
    Analyze URL and domain for phishing indicators using GNN models
    """
    start_time = time.time()
    
    try:
        # Use provided domain or extract from URL
        domain = request.domain
        if not domain:
            parsed = urlparse(request.url)
            domain = parsed.netloc or parsed.path.split('/')[0]
        
        # Extract features
        features = extract_url_features(request.url)
        domain_info = features.get("domain_info", {})
        
        # Calculate phishing probability
        phishing_probability = calculate_phishing_score(features)
        
        # Determine if phishing (threshold: 0.5)
        is_phishing = phishing_probability >= 0.5
        
        processing_time = (time.time() - start_time) * 1000
        
        return URLAnalysisResponse(
            is_phishing=is_phishing,
            confidence=phishing_probability,
            features=features,
            model_version="1.0.0-rule-based",
            domain_info=domain_info,
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
