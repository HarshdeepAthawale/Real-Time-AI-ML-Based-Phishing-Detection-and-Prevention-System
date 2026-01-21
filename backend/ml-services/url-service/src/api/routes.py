from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    URLAnalysisRequest, 
    DomainAnalysisRequest,
    RedirectChainRequest,
    DomainReputationRequest,
    URLAnalysisResponse,
    DomainAnalysisResponse,
    RedirectChainResponse,
    DomainReputationResponse,
    CompatibilityAnalysisRequest,
    CompatibilityAnalysisResponse
)
from src.analyzers.url_parser import URLParser
from src.analyzers.domain_analyzer import DomainAnalyzer
from src.analyzers.homoglyph_detector import HomoglyphDetector
from src.analyzers.dns_analyzer import DNSAnalyzer
from src.analyzers.whois_analyzer import WHOISAnalyzer
from src.analyzers.ssl_analyzer import SSLAnalyzer
from src.crawler.redirect_tracker import RedirectTracker
from src.graph.graph_builder import GraphBuilder
from src.models.gnn_classifier import DomainGNNClassifier
from src.models.reputation_scorer import ReputationScorer
from src.utils.cache import Cache
from src.utils.validators import URLValidator
import time
import logging
import os
import torch

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize analyzers (singleton pattern)
url_parser = URLParser()
domain_analyzer = DomainAnalyzer()
homoglyph_detector = HomoglyphDetector()
dns_analyzer = DNSAnalyzer()
whois_analyzer = WHOISAnalyzer()
ssl_analyzer = SSLAnalyzer()
redirect_tracker = RedirectTracker()
graph_builder = GraphBuilder()
reputation_scorer = ReputationScorer()
url_validator = URLValidator()

# Initialize cache
try:
    cache = Cache()
    logger.info("Cache initialized")
except Exception as e:
    logger.warning(f"Cache initialization failed: {e}. Caching will be disabled.")
    cache = None

# Initialize GNN model (lazy loading)
gnn_model = None
gnn_model_loaded = False

def get_gnn_model():
    """Lazy load GNN model with improved error handling"""
    global gnn_model, gnn_model_loaded
    
    if gnn_model_loaded:
        return gnn_model
    
    try:
        # Determine device
        device = os.getenv("DEVICE", "cpu")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get model path from environment or use default
        model_path = os.getenv("GNN_MODEL_PATH", "./models/gnn-domain-classifier-v1/model.pt")
        model_dir = os.path.dirname(model_path)
        
        logger.info(f"Loading GNN model from {model_path} on device: {device}")
        
            # Initialize model
        gnn_model = DomainGNNClassifier(input_dim=5, hidden_dim=64, num_classes=2)
        
        # Move to device
        if device != "cpu":
            try:
                gnn_model = gnn_model.to(device)
            except Exception as e:
                logger.warning(f"Failed to move model to {device}, using CPU: {e}")
                device = "cpu"
        
        # Try to load trained weights
        if os.path.exists(model_path):
            try:
                gnn_model.load(model_path, device=device)
                logger.info(f"✅ GNN model loaded successfully from {model_path}")
                gnn_model_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load model weights from {model_path}: {e}")
                logger.info("Using untrained model (random weights)")
                gnn_model_loaded = True  # Still mark as loaded, but with untrained weights
        else:
            logger.info(f"⚠️  GNN model not found at {model_path}")
            logger.info("Using untrained model (random weights)")
            logger.info("To train a model, run: python training/train_gnn_model.py")
            gnn_model_loaded = True  # Mark as loaded even without trained weights
        
        # Set to evaluation mode
        gnn_model.eval()
        
    except Exception as e:
        logger.error(f"Failed to initialize GNN model: {e}", exc_info=True)
        gnn_model = None
        gnn_model_loaded = False
    
    return gnn_model

async def perform_url_analysis(url: str, legitimate_domain: str = None):
    """Helper function to perform URL analysis - can be reused by multiple endpoints"""
    # Validate URL
    if not url_validator.is_valid_url(url):
        sanitized = url_validator.sanitize_url(url)
        if not sanitized:
            raise HTTPException(status_code=400, detail="Invalid URL format")
        url = sanitized
    
    # Check cache
    cache_key = f"url_analysis:{hash(url)}"
    if cache and cache.exists(cache_key):
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Returning cached result for URL: {url}")
            return cached_result
    
    # Parse URL
    parsed_url = url_parser.parse(url)
    
    # Analyze domain
    domain_analysis = domain_analyzer.analyze(parsed_url["registered_domain"])
    
    # DNS analysis
    dns_analysis = dns_analyzer.analyze(parsed_url["registered_domain"])
    
    # WHOIS analysis
    whois_analysis = whois_analyzer.analyze(parsed_url["registered_domain"])
    
    # SSL analysis (only for HTTPS URLs)
    ssl_analysis = None
    if parsed_url["scheme"] == "https":
        try:
            ssl_analysis = ssl_analyzer.analyze(parsed_url["registered_domain"])
        except Exception as e:
            logger.warning(f"SSL analysis failed: {e}")
    
    # Redirect tracking
    redirect_analysis = redirect_tracker.track(url)
    
    # Homoglyph detection (if legitimate domain provided)
    homoglyph_analysis = None
    if legitimate_domain:
        homoglyph_analysis = homoglyph_detector.detect(
            parsed_url["registered_domain"],
            legitimate_domain
        )
    
    # Calculate reputation score
    analysis_results = {
        "whois_analysis": whois_analysis,
        "dns_analysis": dns_analysis,
        "ssl_analysis": ssl_analysis,
        "redirect_analysis": redirect_analysis,
        "homoglyph_analysis": homoglyph_analysis,
        "domain_analysis": domain_analysis
    }
    reputation_score = reputation_scorer.calculate_reputation(analysis_results)
    
    # Try GNN-based prediction if model is available
    gnn_prediction = None
    try:
        model = get_gnn_model()
        if model is not None:
            # Build a simple graph for this domain
            # In production, this would include relationships from database
            domains = [{
                "id": parsed_url["registered_domain"],
                "domain": parsed_url["registered_domain"],
                "reputation_score": reputation_score.get("score", 50.0),
                "age_days": whois_analysis.get("age_days", 0),
                "is_malicious": False,
                "is_suspicious": domain_analysis.get("is_suspicious", False) or 
                                whois_analysis.get("is_suspicious", False)
            }]
            
            # Build graph (may be empty if no relationships)
            graph_data = graph_builder.build_domain_graph(domains, [])
            
            # Make prediction
            if graph_data.x.size(0) > 0:  # Only if graph has nodes
                gnn_prediction = model.predict(graph_data)
                logger.debug(f"GNN prediction: {gnn_prediction}")
    except Exception as e:
        logger.debug(f"GNN prediction skipped: {e}")
    
    result = {
        "url_analysis": parsed_url,
        "domain_analysis": domain_analysis,
        "dns_analysis": dns_analysis,
        "whois_analysis": whois_analysis,
        "ssl_analysis": ssl_analysis,
        "redirect_analysis": redirect_analysis,
        "homoglyph_analysis": homoglyph_analysis,
        "reputation_score": reputation_score,
        "gnn_prediction": gnn_prediction  # Add GNN prediction to result
    }
    
    # Enhance phishing probability with GNN if available
    if gnn_prediction and "malicious_probability" in gnn_prediction:
        # Combine reputation score with GNN prediction
        gnn_malicious_prob = gnn_prediction["malicious_probability"]
        # Use GNN prediction to adjust reputation if confidence is high
        if gnn_prediction.get("confidence", 0) > 0.6:
            # Weighted combination: 70% GNN, 30% rule-based reputation
            combined_score = (gnn_malicious_prob * 0.7) + ((100 - reputation_score.get("score", 50)) / 100 * 0.3)
            result["phishing_probability"] = combined_score
        else:
            # Use reputation-based score if GNN confidence is low
            result["phishing_probability"] = (100 - reputation_score.get("score", 50)) / 100
    else:
        # Fallback to reputation-based score
        result["phishing_probability"] = (100 - reputation_score.get("score", 50)) / 100
    
    # Cache result (1 hour TTL)
    if cache:
        cache.set(cache_key, result, ttl=3600)
    
    return result

@router.post("/analyze-url", response_model=URLAnalysisResponse)
async def analyze_url(request: URLAnalysisRequest):
    """Analyze URL for phishing indicators"""
    start_time = time.time()
    
    try:
        result = await perform_url_analysis(request.url, request.legitimate_domain)
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = round(processing_time, 2)
        return URLAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"URL analysis failed: {str(e)}")

@router.post("/analyze-domain", response_model=DomainAnalysisResponse)
async def analyze_domain(request: DomainAnalysisRequest):
    """Analyze domain using GNN and graph analysis"""
    start_time = time.time()
    
    try:
        # Check cache
        cache_key = f"domain_analysis:{request.domain}"
        if cache and cache.exists(cache_key):
            cached_result = cache.get(cache_key)
            if cached_result:
                return DomainAnalysisResponse(**cached_result)
        
        # Basic domain analysis
        domain_analysis = domain_analyzer.analyze(request.domain)
        dns_analysis = dns_analyzer.analyze(request.domain)
        whois_analysis = whois_analyzer.analyze(request.domain)
        
        analysis = {
            "domain_analysis": domain_analysis,
            "dns_analysis": dns_analysis,
            "whois_analysis": whois_analysis
        }
        
        graph_analysis = None
        gnn_prediction = None
        
        if request.include_graph:
            # Build graph (simplified - would need relationship data from database)
            # For now, create a simple graph with just the domain
            domains = [{
                "id": request.domain,
                "domain": request.domain,
                "reputation_score": 50.0,
                "age_days": whois_analysis.get("age_days", 0),
                "is_malicious": False,
                "is_suspicious": whois_analysis.get("is_suspicious", False)
            }]
            
            relationships = []  # Would come from database
            
            # Build graph
            graph_data = graph_builder.build_domain_graph(domains, relationships)
            
            # Extract graph features
            from src.graph.graph_features import GraphFeatureExtractor
            feature_extractor = GraphFeatureExtractor()
            nx_graph = graph_builder.build_networkx_graph(domains, relationships)
            graph_features = feature_extractor.extract_features(nx_graph, request.domain)
            global_features = feature_extractor.extract_global_features(nx_graph)
            
            graph_analysis = {
                "node_features": graph_features,
                "global_features": global_features
            }
            
            # GNN prediction
            try:
                model = get_gnn_model()
                if model:
                    gnn_prediction = model.predict(graph_data)
            except Exception as e:
                logger.warning(f"GNN prediction failed: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "domain": request.domain,
            "analysis": analysis,
            "graph_analysis": graph_analysis,
            "gnn_prediction": gnn_prediction,
            "processing_time_ms": round(processing_time, 2)
        }
        
        # Cache result
        if cache:
            cache.set(cache_key, result, ttl=3600)
        
        return DomainAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Domain analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Domain analysis failed: {str(e)}")

@router.post("/check-redirect-chain", response_model=RedirectChainResponse)
async def check_redirect_chain(request: RedirectChainRequest):
    """Check redirect chain for URL"""
    try:
        if not url_validator.is_valid_url(request.url):
            sanitized = url_validator.sanitize_url(request.url)
            if not sanitized:
                raise HTTPException(status_code=400, detail="Invalid URL format")
            request.url = sanitized
        
        result = redirect_tracker.track(request.url)
        return RedirectChainResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Redirect tracking failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Redirect tracking failed: {str(e)}")

@router.post("/domain-reputation", response_model=DomainReputationResponse)
async def get_domain_reputation(request: DomainReputationRequest):
    """Get domain reputation score"""
    try:
        # Get all analysis data
        domain_analysis = domain_analyzer.analyze(request.domain)
        dns_analysis = dns_analyzer.analyze(request.domain)
        whois_analysis = whois_analyzer.analyze(request.domain)
        
        try:
            ssl_analysis = ssl_analyzer.analyze(request.domain)
        except:
            ssl_analysis = None
        
        analysis_results = {
            "domain_analysis": domain_analysis,
            "dns_analysis": dns_analysis,
            "whois_analysis": whois_analysis,
            "ssl_analysis": ssl_analysis
        }
        
        reputation = reputation_scorer.calculate_reputation(analysis_results)
        
        return DomainReputationResponse(
            domain=request.domain,
            **reputation
        )
    except Exception as e:
        logger.error(f"Reputation calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reputation calculation failed: {str(e)}")
