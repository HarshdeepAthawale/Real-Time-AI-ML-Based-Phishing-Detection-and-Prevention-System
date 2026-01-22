"""Pydantic request/response models"""
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict


class URLAnalysisRequest(BaseModel):
    """Request model for URL analysis"""
    url: str = Field(..., description="URL to analyze")
    include_dns: bool = Field(default=True, description="Include DNS analysis")
    include_whois: bool = Field(default=True, description="Include WHOIS analysis")
    include_ssl: bool = Field(default=True, description="Include SSL analysis")
    track_redirects: bool = Field(default=True, description="Track redirect chain")
    check_homoglyph: bool = Field(default=True, description="Check for homoglyph attacks")


class DomainAnalysisRequest(BaseModel):
    """Request model for domain analysis"""
    domain: str = Field(..., description="Domain to analyze")
    use_gnn: bool = Field(default=True, description="Use GNN model for classification")


class RedirectCheckRequest(BaseModel):
    """Request model for redirect chain checking"""
    url: str = Field(..., description="URL to check redirects")


class HomoglyphCheckRequest(BaseModel):
    """Request model for homoglyph detection"""
    domain: str = Field(..., description="Domain to check")
    legitimate_domain: Optional[str] = Field(None, description="Legitimate domain to compare against")


class URLAnalysisResponse(BaseModel):
    """Response model for URL analysis"""
    url: str
    is_malicious: bool
    malicious_probability: float
    confidence: float
    url_components: Dict
    domain_analysis: Optional[Dict] = None
    dns_analysis: Optional[Dict] = None
    whois_analysis: Optional[Dict] = None
    ssl_analysis: Optional[Dict] = None
    redirect_analysis: Optional[Dict] = None
    homoglyph_analysis: Optional[Dict] = None
    reputation_score: Optional[float] = None
    risk_level: Optional[str] = None
    processing_time_ms: float
    cached: bool = False


class DomainAnalysisResponse(BaseModel):
    """Response model for domain analysis"""
    domain: str
    malicious_probability: float
    legitimate_probability: float
    confidence: float
    prediction: str
    domain_features: Dict
    reputation_score: float
    risk_level: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    gnn_model_loaded: bool
