"""
NLP Service - Phishing Detection using BERT/RoBERTa models
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NLP Phishing Detection Service", version="1.0.0")


class TextAnalysisRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    include_features: Optional[bool] = False


class TextAnalysisResponse(BaseModel):
    is_phishing: bool
    confidence: float
    features: Dict
    model_version: str
    processing_time_ms: Optional[float] = None


# Phishing indicators (rule-based detection)
PHISHING_KEYWORDS = [
    "urgent", "verify", "suspended", "expired", "click here", "act now",
    "limited time", "verify account", "security alert", "unauthorized access",
    "confirm identity", "update payment", "account locked", "immediate action"
]

URGENCY_PATTERNS = [
    r"within \d+ (hours?|days?)",
    r"expires? (today|tomorrow)",
    r"act (now|immediately|urgently)",
    r"time.?sensitive"
]


def extract_features(text: str) -> Dict:
    """Extract features from text for phishing detection"""
    text_lower = text.lower()
    
    # Keyword matches
    keyword_matches = sum(1 for keyword in PHISHING_KEYWORDS if keyword in text_lower)
    
    # Urgency patterns
    urgency_score = sum(1 for pattern in URGENCY_PATTERNS if re.search(pattern, text_lower, re.IGNORECASE))
    
    # Suspicious patterns
    suspicious_patterns = {
        "has_urls": bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        "has_email": bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        "has_phone": bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)),
        "exclamation_count": text.count('!'),
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        "keyword_matches": keyword_matches,
        "urgency_score": urgency_score
    }
    
    return suspicious_patterns


def calculate_phishing_score(features: Dict) -> float:
    """Calculate phishing probability based on features"""
    score = 0.0
    
    # Keyword matches contribute to score
    score += features.get("keyword_matches", 0) * 0.15
    
    # Urgency patterns
    score += features.get("urgency_score", 0) * 0.2
    
    # Excessive exclamation marks
    if features.get("exclamation_count", 0) > 3:
        score += 0.15
    
    # High uppercase ratio (shouting)
    if features.get("uppercase_ratio", 0) > 0.3:
        score += 0.1
    
    # Has URLs (potential phishing link)
    if features.get("has_urls", False):
        score += 0.2
    
    # Has email (potential spoofing)
    if features.get("has_email", False):
        score += 0.1
    
    # Cap at 1.0
    return min(score, 1.0)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "nlp-service"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "nlp-service", "version": "1.0.0"}


@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text content for phishing indicators using NLP models
    """
    start_time = time.time()
    
    try:
        # Extract features
        features = extract_features(request.text)
        
        # Calculate phishing probability
        phishing_probability = calculate_phishing_score(features)
        
        # Determine if phishing (threshold: 0.5)
        is_phishing = phishing_probability >= 0.5
        
        processing_time = (time.time() - start_time) * 1000
        
        return TextAnalysisResponse(
            is_phishing=is_phishing,
            confidence=phishing_probability,
            features=features if request.include_features else {},
            model_version="1.0.0-rule-based",
            processing_time_ms=round(processing_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
