from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv
from src.api.routes import router
from src.api.schemas import HealthResponse, CompatibilityAnalysisRequest, CompatibilityAnalysisResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("URL Service starting up...")
    try:
        # Initialize any startup tasks here
        logger.info("URL Service started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
    
    yield
    
    # Shutdown
    logger.info("URL Service shutting down...")

app = FastAPI(
    title="URL/Domain Analysis Service",
    description="Phishing detection using Graph Neural Networks and domain analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.post("/analyze", response_model=CompatibilityAnalysisResponse)
async def analyze_compatibility(request: CompatibilityAnalysisRequest):
    """Compatibility endpoint for detection-api - accepts {url} and returns {is_phishing, confidence}"""
    try:
        # Use the shared analysis function
        from src.api.routes import perform_url_analysis
        
        # Get the full analysis result
        analysis_result = await perform_url_analysis(request.url)
        
        # Extract reputation score information
        reputation = analysis_result.get("reputation_score", {}) or {}
        is_suspicious = reputation.get("is_suspicious", False)
        is_malicious = reputation.get("is_malicious", False)
        reputation_score = reputation.get("reputation_score", 50.0)
        
        # Determine is_phishing (true if suspicious or malicious)
        is_phishing = is_suspicious or is_malicious
        
        # Normalize confidence: reputation_score is 0-100, convert to 0-1
        # Lower reputation = higher phishing confidence
        # If reputation < 50, confidence increases as reputation decreases
        if reputation_score < 50:
            confidence = (50 - reputation_score) / 50.0  # 0-1 scale
        else:
            confidence = 0.0
        
        # Extract key features
        whois_analysis = analysis_result.get("whois_analysis", {})
        redirect_analysis = analysis_result.get("redirect_analysis", {})
        homoglyph_analysis = analysis_result.get("homoglyph_analysis")
        ssl_analysis = analysis_result.get("ssl_analysis")
        
        features = {
            "domain_age_days": whois_analysis.get("age_days") if whois_analysis else None,
            "redirect_count": redirect_analysis.get("redirect_count", 0) if redirect_analysis else 0,
            "is_suspicious_redirect": redirect_analysis.get("is_suspicious", False) if redirect_analysis else False,
            "has_homoglyph": homoglyph_analysis is not None and homoglyph_analysis.get("has_homoglyph", False) if homoglyph_analysis else False,
            "ssl_valid": ssl_analysis.get("is_valid", False) if ssl_analysis else None,
            "reputation_level": reputation.get("reputation_level", "unknown")
        }
        
        return CompatibilityAnalysisResponse(
            is_phishing=is_phishing,
            confidence=min(max(confidence, 0.0), 1.0),  # Clamp between 0 and 1
            features=features,
            model_version="1.0.0"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compatibility analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = os.path.exists(os.getenv("GNN_MODEL_PATH", "./models/gnn-domain-classifier-v1/model.pt"))
    return HealthResponse(
        status="healthy",
        service="url-service",
        models_loaded=model_loaded
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "url-service",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
