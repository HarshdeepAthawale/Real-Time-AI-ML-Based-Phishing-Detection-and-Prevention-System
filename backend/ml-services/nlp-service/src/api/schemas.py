"""Pydantic request/response models"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis"""
    text: str = Field(..., description="Text content to analyze")
    include_features: bool = Field(default=False, description="Include detailed features in response")


class EmailAnalysisRequest(BaseModel):
    """Request model for email analysis"""
    raw_email: Optional[str] = Field(None, description="Raw email content (RFC 822 format)")
    subject: Optional[str] = Field(None, description="Email subject")
    body: Optional[str] = Field(None, description="Email body")
    sender: Optional[str] = Field(None, description="Sender email address")
    include_features: bool = Field(default=False, description="Include detailed features in response")


class AIContentDetectionRequest(BaseModel):
    """Request model for AI-generated content detection"""
    text: str = Field(..., description="Text to check for AI generation")


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    phishing_probability: float = Field(..., description="Probability of being phishing (0-1)")
    legitimate_probability: float = Field(..., description="Probability of being legitimate (0-1)")
    confidence: float = Field(..., description="Model confidence (0-1)")
    prediction: str = Field(..., description="Classification result")
    urgency_score: Optional[float] = Field(None, description="Urgency score (0-100)")
    sentiment: Optional[str] = Field(None, description="Sentiment classification")
    social_engineering_score: Optional[float] = Field(None, description="Social engineering score (0-100)")
    ai_generated_probability: Optional[float] = Field(None, description="Probability of being AI-generated")
    features: Optional[Dict] = Field(None, description="Detailed features and indicators")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    cached: bool = Field(default=False, description="Whether result was cached")


class AIDetectionResponse(BaseModel):
    """Response model for AI content detection"""
    ai_generated_probability: float
    human_written_probability: float
    is_ai_generated: bool
    confidence: float
    model_loaded: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    models_loaded: bool


class ModelInfoResponse(BaseModel):
    """Model information response"""
    phishing_classifier: Dict
    ai_detector: Dict
