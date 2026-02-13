"""API route handlers"""
import time
from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    URLAnalysisRequest, DomainAnalysisRequest, RedirectCheckRequest, HomoglyphCheckRequest,
    URLAnalysisResponse, DomainAnalysisResponse, HealthResponse
)
from src.models.gnn_classifier import DomainGNNClassifier
from src.models.reputation_scorer import ReputationScorer
from src.analyzers.url_parser import URLParser
from src.analyzers.domain_analyzer import DomainAnalyzer
from src.analyzers.dns_analyzer import DNSAnalyzer
from src.analyzers.whois_analyzer import WHOISAnalyzer
from src.analyzers.ssl_analyzer import SSLAnalyzer
from src.analyzers.homoglyph_detector import HomoglyphDetector
from src.analyzers.obfuscation_analyzer import ObfuscationAnalyzer
from src.crawler.redirect_tracker import RedirectTracker
from src.graph.graph_builder import GraphBuilder
from src.utils.cache import cache_service
from src.utils.logger import logger
from src.config import settings

router = APIRouter()

# Initialize analyzers and models
url_parser = URLParser()
domain_analyzer = DomainAnalyzer()
dns_analyzer = DNSAnalyzer()
whois_analyzer = WHOISAnalyzer()
ssl_analyzer = SSLAnalyzer()
homoglyph_detector = HomoglyphDetector()
obfuscation_analyzer = ObfuscationAnalyzer()
redirect_tracker = RedirectTracker()
graph_builder = GraphBuilder()
reputation_scorer = ReputationScorer()

# Initialize GNN model
gnn_classifier = DomainGNNClassifier(
    model_path=settings.gnn_model_path,
    device=settings.inference_device
)


def _compute_phishing_probability(
    reputation_result: dict,
    obfuscation_analysis: dict,
    redirect_analysis: dict | None,
    homoglyph_analysis: dict | None,
    gnn_probability: float,
) -> float:
    """Compute an ensemble phishing probability from heuristic signals."""
    score = 0.0
    weight_total = 0.0

    # Reputation score (inverted: low reputation = high phishing probability)
    rep_score = reputation_result.get("reputation_score", 100)
    rep_phishing = max(0.0, (100 - rep_score) / 100.0)
    score += rep_phishing * 0.30
    weight_total += 0.30

    # Obfuscation score (strengthened - key indicator for phishing URLs)
    obf_score = obfuscation_analysis.get("obfuscation_score", 0.0)
    obf_techniques = len(obfuscation_analysis.get("techniques_detected", []))
    # Base obfuscation + bonus for multiple techniques detected
    obf_contribution = min(1.0, obf_score * 1.1 + obf_techniques * 0.05)
    score += obf_contribution * 0.30
    weight_total += 0.30

    # Redirect analysis
    if redirect_analysis:
        redirect_count = redirect_analysis.get("redirect_count", 0)
        is_suspicious = redirect_analysis.get("is_suspicious", False)
        redirect_signal = min(1.0, redirect_count / 5.0) if redirect_count > 1 else 0.0
        if is_suspicious:
            redirect_signal = max(redirect_signal, 0.6)
        score += redirect_signal * 0.15
        weight_total += 0.15

    # Homoglyph analysis
    if homoglyph_analysis:
        is_homoglyph = homoglyph_analysis.get("is_suspicious", False)
        if is_homoglyph:
            score += 0.15
        weight_total += 0.15

    # GNN model (if available and not default 0.5)
    if abs(gnn_probability - 0.5) > 0.01:
        score += gnn_probability * 0.15
        weight_total += 0.15

    return min(1.0, score / weight_total) if weight_total > 0 else 0.0


@router.post("/analyze-url", response_model=URLAnalysisResponse)
async def analyze_url(request: URLAnalysisRequest):
    """Complete URL analysis"""
    start_time = time.time()
    
    # Check cache
    cache_key = cache_service.get_url_cache_key(request.url)
    cached_result = await cache_service.get(cache_key)
    if cached_result:
        cached_result["cached"] = True
        return cached_result
    
    try:
        # Parse URL
        url_components = url_parser.parse(request.url)
        
        if "error" in url_components:
            raise HTTPException(status_code=400, detail=f"Invalid URL: {url_components['error']}")
        
        domain = url_components.get("registered_domain", "")
        
        # Domain analysis
        domain_analysis = domain_analyzer.analyze(domain)
        
        # DNS analysis
        dns_analysis = None
        if request.include_dns:
            dns_analysis = dns_analyzer.analyze(domain)
        
        # WHOIS analysis
        whois_analysis = None
        if request.include_whois:
            whois_analysis = whois_analyzer.analyze(domain)
            if whois_analysis.get("age_days"):
                domain_analysis["age_days"] = whois_analysis["age_days"]
        
        # SSL analysis
        ssl_analysis = None
        if request.include_ssl:
            ssl_analysis = ssl_analyzer.analyze(domain)
            domain_analysis["has_valid_ssl"] = ssl_analysis.get("is_valid", False)
        
        # Redirect tracking
        redirect_analysis = None
        if request.track_redirects:
            redirect_analysis = redirect_tracker.track(request.url)
        
        # Homoglyph detection
        homoglyph_analysis = None
        if request.check_homoglyph:
            homoglyph_analysis = homoglyph_detector.check_against_popular_domains(domain)

        # Obfuscation analysis
        obfuscation_analysis = obfuscation_analyzer.analyze(request.url)

        # Calculate reputation score
        if dns_analysis:
            domain_analysis["has_dns"] = dns_analysis.get("has_dns", False)

        reputation_result = reputation_scorer.calculate_score(domain_analysis)

        # GNN prediction (if model is loaded)
        malicious_probability = 0.5  # Default
        if gnn_classifier.model:
            # Build simple graph for single domain
            graph_data = graph_builder.build_domain_graph([{
                **domain_analysis,
                "domain": domain,
                "reputation_score": reputation_result["reputation_score"]
            }])

            gnn_result = gnn_classifier.predict(graph_data)
            malicious_probability = gnn_result.get("malicious_probability", 0.5)

        # Compute heuristic phishing probability from all signals
        phishing_probability = _compute_phishing_probability(
            reputation_result, obfuscation_analysis,
            redirect_analysis, homoglyph_analysis,
            malicious_probability
        )

        # Composite is_suspicious: obfuscation, homoglyph, suspicious redirect, or high heuristic score
        is_suspicious = (
            obfuscation_analysis.get("is_obfuscated", False)
            or (homoglyph_analysis or {}).get("is_suspicious", False)
            or (redirect_analysis or {}).get("is_suspicious", False)
            or reputation_result.get("risk_level", "low") in ("high", "critical")
            or phishing_probability >= 0.5
        )

        processing_time = (time.time() - start_time) * 1000

        response = URLAnalysisResponse(
            url=request.url,
            is_malicious=phishing_probability > 0.5,
            is_suspicious=is_suspicious,
            malicious_probability=malicious_probability,
            confidence=abs(phishing_probability - 0.5) * 2,
            url_components=url_components,
            domain_analysis=domain_analysis,
            dns_analysis=dns_analysis,
            whois_analysis=whois_analysis,
            ssl_analysis=ssl_analysis,
            redirect_analysis=redirect_analysis,
            homoglyph_analysis=homoglyph_analysis,
            obfuscation_analysis=obfuscation_analysis,
            reputation_score=reputation_result["reputation_score"],
            risk_level=reputation_result["risk_level"],
            phishing_probability=phishing_probability,
            processing_time_ms=processing_time,
            cached=False
        )
        
        # Cache result
        await cache_service.set(cache_key, response.dict())
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-domain", response_model=DomainAnalysisResponse)
async def analyze_domain(request: DomainAnalysisRequest):
    """Analyze domain using GNN"""
    start_time = time.time()
    
    # Check cache
    cache_key = cache_service.get_domain_cache_key(request.domain)
    cached_result = await cache_service.get(cache_key)
    if cached_result:
        return cached_result
    
    try:
        # Domain analysis
        domain_features = domain_analyzer.analyze(request.domain)
        
        # Calculate reputation
        reputation_result = reputation_scorer.calculate_score(domain_features)
        
        # GNN prediction
        malicious_prob = 0.5
        legitimate_prob = 0.5
        
        if request.use_gnn and gnn_classifier.model:
            graph_data = graph_builder.build_domain_graph([{
                **domain_features,
                "domain": request.domain,
                "reputation_score": reputation_result["reputation_score"]
            }])
            
            gnn_result = gnn_classifier.predict(graph_data)
            malicious_prob = gnn_result.get("malicious_probability", 0.5)
            legitimate_prob = gnn_result.get("legitimate_probability", 0.5)
        
        processing_time = (time.time() - start_time) * 1000
        
        response = DomainAnalysisResponse(
            domain=request.domain,
            malicious_probability=malicious_prob,
            legitimate_probability=legitimate_prob,
            confidence=abs(malicious_prob - legitimate_prob),
            prediction="malicious" if malicious_prob > 0.5 else "legitimate",
            domain_features=domain_features,
            reputation_score=reputation_result["reputation_score"],
            risk_level=reputation_result["risk_level"],
            processing_time_ms=processing_time
        )
        
        # Cache result
        await cache_service.set(cache_key, response.dict())
        
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing domain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-redirect-chain")
async def check_redirect_chain(request: RedirectCheckRequest):
    """Check HTTP redirect chain"""
    try:
        result = redirect_tracker.track(request.url)
        return result
    except Exception as e:
        logger.error(f"Error checking redirects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-homoglyph")
async def detect_homoglyph(request: HomoglyphCheckRequest):
    """Detect homoglyph attacks"""
    try:
        result = homoglyph_detector.detect(
            request.domain,
            request.legitimate_domain
        )
        
        # Also check against popular domains
        brand_check = homoglyph_detector.check_against_popular_domains(request.domain)
        result["brand_check"] = brand_check
        
        return result
    except Exception as e:
        logger.error(f"Error detecting homoglyphs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
        version=settings.service_version,
        gnn_model_loaded=gnn_classifier.model is not None
    )
