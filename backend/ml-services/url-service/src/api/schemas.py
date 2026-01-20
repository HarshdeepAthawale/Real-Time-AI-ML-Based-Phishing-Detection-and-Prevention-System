from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, List

class URLAnalysisRequest(BaseModel):
    url: str = Field(..., description="URL to analyze")
    legitimate_domain: Optional[str] = Field(None, description="Legitimate domain for comparison (homoglyph detection)")

class DomainAnalysisRequest(BaseModel):
    domain: str = Field(..., description="Domain to analyze")
    include_graph: bool = Field(False, description="Include graph-based analysis")

class RedirectChainRequest(BaseModel):
    url: str = Field(..., description="URL to track redirect chain")

class DomainReputationRequest(BaseModel):
    domain: str = Field(..., description="Domain to get reputation for")

class URLAnalysisResponse(BaseModel):
    url_analysis: Dict
    domain_analysis: Dict
    dns_analysis: Dict
    whois_analysis: Dict
    ssl_analysis: Optional[Dict] = None
    redirect_analysis: Dict
    homoglyph_analysis: Optional[Dict] = None
    reputation_score: Optional[Dict] = None
    processing_time_ms: float

class DomainAnalysisResponse(BaseModel):
    domain: str
    analysis: Dict
    graph_analysis: Optional[Dict] = None
    gnn_prediction: Optional[Dict] = None
    processing_time_ms: float

class RedirectChainResponse(BaseModel):
    original_url: str
    final_url: str
    redirect_count: int
    redirects: List[Dict]
    is_suspicious: bool

class DomainReputationResponse(BaseModel):
    domain: str
    reputation_score: float
    reputation_level: str
    is_suspicious: bool
    is_malicious: bool

class HealthResponse(BaseModel):
    status: str
    service: str
    models_loaded: bool = False

class CompatibilityAnalysisRequest(BaseModel):
    url: str = Field(..., description="URL to analyze")

class CompatibilityAnalysisResponse(BaseModel):
    is_phishing: bool = Field(..., description="Whether the URL is identified as phishing")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    features: Optional[Dict] = Field(None, description="Key analysis features")
    model_version: str = Field(default="1.0.0", description="Model version used")
