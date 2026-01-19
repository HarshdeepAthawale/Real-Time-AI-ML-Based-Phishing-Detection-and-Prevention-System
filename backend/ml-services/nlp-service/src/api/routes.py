from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import (
    TextAnalysisRequest, 
    EmailAnalysisRequest, 
    AnalysisResponse,
    AIDetectionResponse
)
from src.models.phishing_classifier import PhishingClassifier
from src.models.ai_detector import AIGeneratedDetector
from src.preprocessing.text_normalizer import TextNormalizer
from src.preprocessing.email_parser import EmailParser
from src.analyzers.urgency_analyzer import UrgencyAnalyzer
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.social_engineering import SocialEngineeringAnalyzer
from src.models.feature_extractor import FeatureExtractor
from src.utils.model_loader import ModelLoader
from src.utils.cache import RedisCache
import time
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize analyzers (singleton pattern)
text_normalizer = TextNormalizer()
email_parser = EmailParser()
urgency_analyzer = UrgencyAnalyzer()
sentiment_analyzer = SentimentAnalyzer()
social_engineering_analyzer = SocialEngineeringAnalyzer()
feature_extractor = FeatureExtractor()

# Initialize Redis cache (graceful fallback if unavailable)
try:
    cache = RedisCache()
    logger.info("Redis cache initialized")
except Exception as e:
    logger.warning(f"Redis cache initialization failed: {e}. Caching will be disabled.")
    cache = None

def get_models():
    """Dependency to get loaded models"""
    loader = ModelLoader.get_instance()
    if loader is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return loader

@router.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    models: ModelLoader = Depends(get_models)
):
    """
    Analyze text for phishing indicators.
    
    This endpoint performs comprehensive analysis including:
    - Phishing probability detection using transformer models
    - AI-generated content detection
    - Urgency score calculation
    - Sentiment analysis
    - Social engineering indicators
    
    Results are cached for 1 hour to improve performance.
    """
    start_time = time.time()
    
    try:
        # Check cache first
        if cache:
            cached_result = cache.get_analysis(request.text)
            if cached_result:
                logger.debug("Returning cached result for text analysis")
                return AnalysisResponse(**cached_result)
        
        # Normalize text
        normalized_text = text_normalizer.normalize(request.text)
        
        # Phishing detection
        phishing_result = models.phishing_classifier.predict(normalized_text)
        
        # AI detection
        ai_result = models.ai_detector.detect(normalized_text)
        
        # Urgency analysis
        urgency_result = urgency_analyzer.analyze(request.text)
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze(request.text)
        
        # Social engineering analysis
        se_result = social_engineering_analyzer.analyze(request.text)
        
        # Extract features if requested
        features = None
        if request.include_features:
            features = models.phishing_classifier.extract_features(normalized_text)
            linguistic_features = feature_extractor.extract_linguistic_features(request.text)
            features.update({
                "linguistic": linguistic_features,
                "urgency": urgency_result,
                "sentiment": sentiment_result,
                "social_engineering": se_result
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        result = AnalysisResponse(
            phishing_probability=phishing_result["phishing_probability"],
            legitimate_probability=phishing_result["legitimate_probability"],
            confidence=phishing_result["confidence"],
            prediction=phishing_result["prediction"],
            ai_generated_probability=ai_result["ai_generated_probability"],
            urgency_score=urgency_result["urgency_score"],
            sentiment=sentiment_result["sentiment"],
            social_engineering_score=se_result["social_engineering_score"],
            features=features,
            processing_time_ms=round(processing_time, 2)
        )
        
        # Cache result
        if cache:
            try:
                result_dict = result.model_dump()
                cache.set_analysis(request.text, result_dict, ttl=3600)
                logger.debug("Cached text analysis result")
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Text analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze-email", response_model=AnalysisResponse)
async def analyze_email(
    request: EmailAnalysisRequest,
    models: ModelLoader = Depends(get_models)
):
    """
    Analyze email for phishing indicators.
    
    Parses raw email content and performs comprehensive analysis including:
    - Email header parsing (From, To, Subject)
    - Body text extraction (text and HTML)
    - Phishing probability detection
    - AI-generated content detection
    - Urgency and sentiment analysis
    - Social engineering indicators
    
    Results are cached for 1 hour to improve performance.
    """
    start_time = time.time()
    
    try:
        # Check cache first (use raw_email as key)
        if cache:
            cached_result = cache.get_analysis(request.raw_email)
            if cached_result:
                logger.debug("Returning cached result for email analysis")
                return AnalysisResponse(**cached_result)
        
        # Parse email
        parsed_email = email_parser.parse(request.raw_email)
        
        # Combine subject and body for analysis
        combined_text = f"{parsed_email['subject']} {parsed_email['body_text']}"
        normalized_text = text_normalizer.normalize(combined_text)
        
        # Phishing detection
        phishing_result = models.phishing_classifier.predict(normalized_text)
        
        # AI detection
        ai_result = models.ai_detector.detect(normalized_text)
        
        # Urgency analysis
        urgency_result = urgency_analyzer.analyze(combined_text)
        
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze(combined_text)
        
        # Social engineering analysis
        se_result = social_engineering_analyzer.analyze(combined_text)
        
        # Extract features if requested
        features = None
        if request.include_features:
            features = models.phishing_classifier.extract_features(normalized_text)
            linguistic_features = feature_extractor.extract_linguistic_features(combined_text)
            features.update({
                "linguistic": linguistic_features,
                "urgency": urgency_result,
                "sentiment": sentiment_result,
                "social_engineering": se_result,
                "email_metadata": {
                    "from": parsed_email.get("from", ""),
                    "subject": parsed_email.get("subject", ""),
                    "has_html": parsed_email.get("body_html") is not None
                }
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        result = AnalysisResponse(
            phishing_probability=phishing_result["phishing_probability"],
            legitimate_probability=phishing_result["legitimate_probability"],
            confidence=phishing_result["confidence"],
            prediction=phishing_result["prediction"],
            ai_generated_probability=ai_result["ai_generated_probability"],
            urgency_score=urgency_result["urgency_score"],
            sentiment=sentiment_result["sentiment"],
            social_engineering_score=se_result["social_engineering_score"],
            features=features,
            processing_time_ms=round(processing_time, 2)
        )
        
        # Cache result
        if cache:
            try:
                result_dict = result.model_dump()
                cache.set_analysis(request.raw_email, result_dict, ttl=3600)
                logger.debug("Cached email analysis result")
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Email analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/detect-ai-content", response_model=AIDetectionResponse)
async def detect_ai_content(
    text: str,
    models: ModelLoader = Depends(get_models)
):
    """
    Detect if content is AI-generated.
    
    Uses a fine-tuned RoBERTa model to distinguish between
    AI-generated and human-written text.
    
    Returns probability scores and confidence level.
    """
    try:
        normalized_text = text_normalizer.normalize(text)
        result = models.ai_detector.detect(normalized_text)
        return AIDetectionResponse(
            ai_generated_probability=result["ai_generated_probability"],
            human_written_probability=result["human_written_probability"],
            is_ai_generated=result["is_ai_generated"],
            confidence=result["confidence"]
        )
    except Exception as e:
        logger.error(f"AI detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI detection failed: {str(e)}")
