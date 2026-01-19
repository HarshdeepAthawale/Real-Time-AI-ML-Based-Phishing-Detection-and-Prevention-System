from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict

class TextAnalysisRequest(BaseModel):
    text: str
    include_features: bool = False

class EmailAnalysisRequest(BaseModel):
    raw_email: str
    include_features: bool = False

class AnalysisResponse(BaseModel):
    phishing_probability: float
    legitimate_probability: float
    confidence: float
    prediction: str
    ai_generated_probability: Optional[float] = None
    urgency_score: Optional[float] = None
    sentiment: Optional[str] = None
    social_engineering_score: Optional[float] = None
    features: Optional[Dict] = None
    processing_time_ms: float

class AIDetectionResponse(BaseModel):
    ai_generated_probability: float
    human_written_probability: float
    is_ai_generated: bool
    confidence: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    service: str = "nlp-service"
