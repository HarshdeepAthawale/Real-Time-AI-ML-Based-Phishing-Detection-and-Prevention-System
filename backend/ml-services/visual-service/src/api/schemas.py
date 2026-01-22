"""Pydantic request/response models"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class PageAnalysisRequest(BaseModel):
    """Request model for page analysis"""
    url: str = Field(..., description="URL to analyze")
    legitimate_url: Optional[str] = Field(None, description="Legitimate URL to compare against")
    include_screenshot: bool = Field(default=True, description="Include screenshot in response")
    include_dom: bool = Field(default=True, description="Include DOM analysis")
    include_brand_detection: bool = Field(default=True, description="Include brand detection")


class DOMAnalysisRequest(BaseModel):
    """Request model for DOM-only analysis"""
    html_content: str = Field(..., description="HTML content to analyze")
    url: Optional[str] = Field(None, description="Source URL (optional)")


class BrandDetectionRequest(BaseModel):
    """Request model for brand detection"""
    url: Optional[str] = Field(None, description="URL to analyze")
    screenshot_bytes: Optional[bytes] = Field(None, description="Screenshot image bytes")
    legitimate_url: Optional[str] = Field(None, description="Legitimate URL to compare")


class VisualCompareRequest(BaseModel):
    """Request model for visual comparison"""
    url1: str = Field(..., description="First URL to compare")
    url2: str = Field(..., description="Second URL to compare")


class ScreenshotAnalysisRequest(BaseModel):
    """Request model for screenshot analysis"""
    screenshot_bytes: bytes = Field(..., description="Screenshot image bytes")
    url: Optional[str] = Field(None, description="Source URL (optional)")


class PageAnalysisResponse(BaseModel):
    """Response model for page analysis"""
    url: str
    is_suspicious: bool
    suspicious_score: float
    confidence: float
    screenshot: Optional[str] = None  # Base64 encoded
    dom_analysis: Optional[Dict] = None
    form_analysis: Optional[Dict] = None
    brand_detection: Optional[Dict] = None
    css_analysis: Optional[Dict] = None
    logo_detection: Optional[Dict] = None
    visual_similarity: Optional[Dict] = None
    processing_time_ms: float
    cached: bool = False


class DOMAnalysisResponse(BaseModel):
    """Response model for DOM analysis"""
    dom_hash: str
    element_count: int
    form_count: int
    link_count: int
    image_count: int
    script_count: int
    forms: List[Dict]
    links: List[Dict]
    structure: Dict
    is_suspicious: bool
    processing_time_ms: float


class BrandDetectionResponse(BaseModel):
    """Response model for brand detection"""
    brand_id: int
    brand_name: Optional[str] = None
    confidence: float
    is_impersonation: bool
    model_loaded: bool
    processing_time_ms: float


class VisualCompareResponse(BaseModel):
    """Response model for visual comparison"""
    similarity_score: float
    is_similar: bool
    differences: List[str]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    cnn_model_loaded: bool
    browser_available: bool
