from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict

class PageAnalysisRequest(BaseModel):
    url: str
    legitimate_url: Optional[str] = None
    wait_time: Optional[int] = 3000

class DOMAnalysisRequest(BaseModel):
    html_content: str

class VisualComparisonRequest(BaseModel):
    url1: str
    url2: str

class PageAnalysisResponse(BaseModel):
    url: str
    dom_analysis: Dict
    form_analysis: Dict
    css_analysis: Optional[Dict] = None
    brand_prediction: Optional[Dict] = None
    similarity_analysis: Optional[Dict] = None
    logo_detection: Optional[Dict] = None
    processing_time_ms: float
    screenshot_url: Optional[str] = None

class DOMAnalysisResponse(BaseModel):
    dom_hash: str
    element_count: int
    forms: List[Dict]
    links: List[Dict]
    images: List[Dict]
    structure: Dict
    is_suspicious: bool

class VisualComparisonResponse(BaseModel):
    similarity_score: float
    hamming_distance: int
    is_similar: bool

class HealthResponse(BaseModel):
    status: str
    service: str = "visual-service"
    models_loaded: bool = False
