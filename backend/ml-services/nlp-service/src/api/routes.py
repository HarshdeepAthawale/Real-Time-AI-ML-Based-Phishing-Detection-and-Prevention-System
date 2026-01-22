"""API route handlers"""
import time
from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import (
    TextAnalysisRequest, EmailAnalysisRequest, AIContentDetectionRequest,
    AnalysisResponse, AIDetectionResponse, HealthResponse, ModelInfoResponse
)
from src.models.model_loader import model_loader
from src.preprocessing.text_normalizer import TextNormalizer
from src.preprocessing.email_parser import EmailParser
from src.preprocessing.feature_extractor import FeatureExtractor
from src.analyzers.urgency_analyzer import UrgencyAnalyzer
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.social_engineering import SocialEngineeringAnalyzer
from src.utils.cache import cache_service
from src.utils.logger import logger
from src.config import settings

router = APIRouter()

# Initialize analyzers (singletons)
text_normalizer = TextNormalizer()
email_parser = EmailParser()
feature_extractor = FeatureExtractor()
urgency_analyzer = UrgencyAnalyzer()
sentiment_analyzer = SentimentAnalyzer()
se_analyzer = SocialEngineeringAnalyzer()


@router.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text for phishing indicators"""
    start_time = time.time()
    
    # Check cache
    cache_key = cache_service.get_text_cache_key(request.text)
    cached_result = await cache_service.get(cache_key)
    if cached_result:
        cached_result["cached"] = True
        return cached_result
    
    try:
        # Normalize text
        normalized_text = text_normalizer.normalize(request.text)
        
        # Phishing detection
        if model_loader.phishing_classifier:
            phishing_result = model_loader.phishing_classifier.predict(normalized_text)
        else:
            phishing_result = {
                "phishing_probability": 0.0,
                "legitimate_probability": 1.0,
                "confidence": 0.0,
                "prediction": "unknown"
            }
        
        # AI detection
        ai_result = {"ai_generated_probability": 0.0}
        if model_loader.ai_detector:
            ai_result = model_loader.ai_detector.detect(normalized_text)
        
        # Urgency analysis
        urgency_result = urgency_analyzer.analyze(request.text)
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze(request.text)
        
        # Social engineering analysis
        se_result = se_analyzer.analyze(request.text)
        
        # Extract features if requested
        features = None
        if request.include_features:
            features = {
                "text_features": feature_extractor.extract_text_features(request.text),
                "urgency": urgency_result,
                "sentiment": sentiment_result,
                "social_engineering": se_result,
                "ai_detection": ai_result
            }
        
        processing_time = (time.time() - start_time) * 1000
        
        response = AnalysisResponse(
            phishing_probability=phishing_result.get("phishing_probability", 0.0),
            legitimate_probability=phishing_result.get("legitimate_probability", 1.0),
            confidence=phishing_result.get("confidence", 0.0),
            prediction=phishing_result.get("prediction", "unknown"),
            ai_generated_probability=ai_result.get("ai_generated_probability"),
            urgency_score=urgency_result.get("urgency_score"),
            sentiment=sentiment_result.get("sentiment"),
            social_engineering_score=se_result.get("social_engineering_score"),
            features=features,
            processing_time_ms=processing_time,
            cached=False
        )
        
        # Cache result
        await cache_service.set(cache_key, response.dict())
        
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-email", response_model=AnalysisResponse)
async def analyze_email(request: EmailAnalysisRequest):
    """Analyze email for phishing indicators"""
    start_time = time.time()
    
    try:
        # Parse email if raw format provided
        if request.raw_email:
            parsed_email = email_parser.parse(request.raw_email)
            subject = parsed_email.get("subject", "")
            body = parsed_email.get("body_text", "")
            sender = parsed_email.get("from", "")
        else:
            subject = request.subject or ""
            body = request.body or ""
            sender = request.sender or ""
            parsed_email = {
                "subject": subject,
                "body_text": body,
                "from": sender
            }
        
        # Combine subject and body for analysis
        combined_text = f"{subject} {body}"
        
        # Check cache
        cache_key = cache_service.get_email_cache_key(combined_text)
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            cached_result["cached"] = True
            return cached_result
        
        # Normalize text
        normalized_text = text_normalizer.normalize(combined_text)
        
        # Phishing detection
        if model_loader.phishing_classifier:
            phishing_result = model_loader.phishing_classifier.predict(normalized_text)
        else:
            phishing_result = {
                "phishing_probability": 0.0,
                "legitimate_probability": 1.0,
                "confidence": 0.0,
                "prediction": "unknown"
            }
        
        # AI detection
        ai_result = {"ai_generated_probability": 0.0}
        if model_loader.ai_detector:
            ai_result = model_loader.ai_detector.detect(normalized_text)
        
        # Urgency analysis
        urgency_result = urgency_analyzer.analyze(combined_text)
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze(combined_text)
        
        # Social engineering analysis
        se_result = se_analyzer.analyze(combined_text)
        
        # Extract features if requested
        features = None
        if request.include_features:
            features = {
                "email_features": feature_extractor.extract_email_features(parsed_email),
                "urgency": urgency_result,
                "sentiment": sentiment_result,
                "social_engineering": se_result,
                "ai_detection": ai_result
            }
        
        processing_time = (time.time() - start_time) * 1000
        
        response = AnalysisResponse(
            phishing_probability=phishing_result.get("phishing_probability", 0.0),
            legitimate_probability=phishing_result.get("legitimate_probability", 1.0),
            confidence=phishing_result.get("confidence", 0.0),
            prediction=phishing_result.get("prediction", "unknown"),
            ai_generated_probability=ai_result.get("ai_generated_probability"),
            urgency_score=urgency_result.get("urgency_score"),
            sentiment=sentiment_result.get("sentiment"),
            social_engineering_score=se_result.get("social_engineering_score"),
            features=features,
            processing_time_ms=processing_time,
            cached=False
        )
        
        # Cache result
        await cache_service.set(cache_key, response.dict())
        
        return response
    
    except Exception as e:
        logger.error(f"Error analyzing email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-ai-content", response_model=AIDetectionResponse)
async def detect_ai_content(request: AIContentDetectionRequest):
    """Detect if content is AI-generated"""
    try:
        if model_loader.ai_detector:
            result = model_loader.ai_detector.detect(request.text)
            return AIDetectionResponse(**result)
        else:
            return AIDetectionResponse(
                ai_generated_probability=0.0,
                human_written_probability=1.0,
                is_ai_generated=False,
                confidence=0.0,
                model_loaded=False
            )
    except Exception as e:
        logger.error(f"Error detecting AI content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
        version=settings.service_version,
        models_loaded=model_loader.is_loaded()
    )


@router.get("/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about loaded models"""
    phishing_info = {
        "loaded": model_loader.phishing_classifier is not None and model_loader.phishing_classifier.model is not None,
        "path": settings.phishing_model_path,
        "device": settings.inference_device
    }
    
    ai_detector_info = {
        "loaded": model_loader.ai_detector is not None and model_loader.ai_detector.model is not None,
        "path": settings.ai_detector_model_path,
        "device": settings.inference_device
    }
    
    return ModelInfoResponse(
        phishing_classifier=phishing_info,
        ai_detector=ai_detector_info
    )
