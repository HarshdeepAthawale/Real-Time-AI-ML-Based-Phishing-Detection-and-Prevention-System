"""API route handlers"""
import time
from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    PageAnalysisRequest, DOMAnalysisRequest, BrandDetectionRequest,
    VisualCompareRequest, ScreenshotAnalysisRequest,
    PageAnalysisResponse, DOMAnalysisResponse, BrandDetectionResponse,
    VisualCompareResponse, HealthResponse
)
from src.models.model_loader import model_loader
from src.renderer.page_renderer import page_renderer
from src.analyzers.dom_analyzer import DOMAnalyzer
from src.analyzers.form_analyzer import FormAnalyzer
from src.analyzers.css_analyzer import CSSAnalyzer
from src.analyzers.logo_detector import LogoDetector
from src.models.visual_similarity import VisualSimilarity
from src.utils.cache import cache_service
from src.utils.logger import logger
from src.config import settings

router = APIRouter()

# Initialize analyzers
dom_analyzer = DOMAnalyzer()
form_analyzer = FormAnalyzer()
css_analyzer = CSSAnalyzer()
logo_detector = LogoDetector()
visual_similarity = VisualSimilarity()


@router.post("/analyze-page", response_model=PageAnalysisResponse)
async def analyze_page(request: PageAnalysisRequest):
    """Complete webpage analysis"""
    start_time = time.time()
    
    # Check cache
    cache_key = cache_service.get_url_cache_key(request.url)
    cached_result = await cache_service.get(cache_key)
    if cached_result:
        cached_result["cached"] = True
        return cached_result
    
    try:
        # Render page
        render_result = await page_renderer.render_page(request.url)
        
        if not render_result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=f"Failed to render page: {render_result.get('error', 'Unknown error')}"
            )
        
        screenshot_bytes = render_result.get("screenshot_bytes")
        dom_content = render_result.get("dom", "")
        
        # DOM analysis
        dom_analysis = None
        form_analysis = None
        if request.include_dom:
            dom_analysis = dom_analyzer.analyze(dom_content)
            if dom_analysis.get("forms"):
                form_analysis = form_analyzer.analyze(dom_analysis["forms"])
        
        # CSS analysis
        css_analysis = None
        if request.include_dom:
            css_analysis = css_analyzer.analyze(dom_content)
        
        # Brand detection
        brand_detection = None
        logo_detection_result = None
        if request.include_brand_detection and screenshot_bytes:
            # Logo detection
            logo_detection_result = logo_detector.detect(screenshot_bytes)
            
            # Brand classification
            if model_loader.brand_classifier and model_loader.brand_classifier.model:
                brand_result = model_loader.brand_classifier.predict(screenshot_bytes)
                brand_detection = {
                    "brand_id": brand_result.get("brand_id", -1),
                    "confidence": brand_result.get("confidence", 0.0),
                    "is_impersonation": brand_result.get("is_impersonation", False),
                    "model_loaded": brand_result.get("model_loaded", False)
                }
        
        # Visual similarity (if legitimate URL provided)
        visual_similarity_result = None
        if request.legitimate_url:
            try:
                legitimate_render = await page_renderer.render_page(request.legitimate_url)
                if legitimate_render.get("success") and legitimate_render.get("screenshot_bytes"):
                    similarity = visual_similarity.compare(
                        screenshot_bytes,
                        legitimate_render["screenshot_bytes"]
                    )
                    visual_similarity_result = similarity
            except Exception as e:
                logger.warning(f"Visual similarity comparison failed: {e}")
        
        # Calculate suspicious score
        suspicious_score = 0.0
        if form_analysis:
            suspicious_score += form_analysis.get("credential_harvesting_score", 0) * 0.4
        if dom_analysis:
            suspicious_score += (30.0 if dom_analysis.get("is_suspicious") else 0.0) * 0.3
        if brand_detection:
            suspicious_score += (brand_detection.get("is_impersonation", False) * 50.0) * 0.3
        
        suspicious_score = min(100.0, suspicious_score)
        is_suspicious = suspicious_score > 50.0
        
        processing_time = (time.time() - start_time) * 1000
        
        # Prepare screenshot (base64 if requested)
        screenshot_base64 = None
        if request.include_screenshot and screenshot_bytes:
            import base64
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode()
        
        response = PageAnalysisResponse(
            url=request.url,
            is_suspicious=is_suspicious,
            suspicious_score=suspicious_score,
            confidence=abs(suspicious_score - 50.0) / 50.0,  # Normalized to 0-1
            screenshot=screenshot_base64,
            dom_analysis=dom_analysis,
            form_analysis=form_analysis,
            brand_detection=brand_detection,
            css_analysis=css_analysis,
            logo_detection=logo_detection_result,
            visual_similarity=visual_similarity_result,
            processing_time_ms=processing_time,
            cached=False
        )
        
        # Cache result
        await cache_service.set(cache_key, response.dict())
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing page: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-dom", response_model=DOMAnalysisResponse)
async def analyze_dom(request: DOMAnalysisRequest):
    """Analyze DOM structure only"""
    start_time = time.time()
    
    try:
        dom_analysis = dom_analyzer.analyze(request.html_content)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DOMAnalysisResponse(
            dom_hash=dom_analysis.get("dom_hash", ""),
            element_count=dom_analysis.get("element_count", 0),
            form_count=dom_analysis.get("form_count", 0),
            link_count=dom_analysis.get("link_count", 0),
            image_count=dom_analysis.get("image_count", 0),
            script_count=dom_analysis.get("script_count", 0),
            forms=dom_analysis.get("forms", []),
            links=dom_analysis.get("links", []),
            structure=dom_analysis.get("structure", {}),
            is_suspicious=dom_analysis.get("is_suspicious", False),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error analyzing DOM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-brand", response_model=BrandDetectionResponse)
async def detect_brand(request: BrandDetectionRequest):
    """Detect brand impersonation"""
    start_time = time.time()
    
    try:
        screenshot_bytes = None
        
        # Get screenshot from URL or use provided bytes
        if request.url and not request.screenshot_bytes:
            render_result = await page_renderer.render_page(request.url)
            if render_result.get("success"):
                screenshot_bytes = render_result.get("screenshot_bytes")
        elif request.screenshot_bytes:
            screenshot_bytes = request.screenshot_bytes
        
        if not screenshot_bytes:
            raise HTTPException(status_code=400, detail="No screenshot available")
        
        # Brand classification
        brand_result = {
            "brand_id": -1,
            "confidence": 0.0,
            "is_impersonation": False,
            "model_loaded": False
        }
        
        if model_loader.brand_classifier and model_loader.brand_classifier.model:
            brand_result = model_loader.brand_classifier.predict(screenshot_bytes)
        
        processing_time = (time.time() - start_time) * 1000
        
        return BrandDetectionResponse(
            brand_id=brand_result.get("brand_id", -1),
            brand_name=None,  # Could be mapped from brand_id
            confidence=brand_result.get("confidence", 0.0),
            is_impersonation=brand_result.get("is_impersonation", False),
            model_loaded=brand_result.get("model_loaded", False),
            processing_time_ms=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting brand: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-visual", response_model=VisualCompareResponse)
async def compare_visual(request: VisualCompareRequest):
    """Compare two URLs visually"""
    start_time = time.time()
    
    try:
        # Render both pages
        render1 = await page_renderer.render_page(request.url1)
        render2 = await page_renderer.render_page(request.url2)
        
        if not render1.get("success") or not render2.get("success"):
            raise HTTPException(
                status_code=400,
                detail="Failed to render one or both pages"
            )
        
        screenshot1 = render1.get("screenshot_bytes")
        screenshot2 = render2.get("screenshot_bytes")
        
        # Compare
        similarity = visual_similarity.compare(screenshot1, screenshot2)
        
        processing_time = (time.time() - start_time) * 1000
        
        return VisualCompareResponse(
            similarity_score=similarity.get("similarity_score", 0.0),
            is_similar=similarity.get("similarity_score", 0.0) > 0.85,
            differences=similarity.get("differences", []),
            processing_time_ms=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing visuals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-screenshot")
async def analyze_screenshot(request: ScreenshotAnalysisRequest):
    """Analyze uploaded screenshot"""
    start_time = time.time()
    
    try:
        # Brand detection
        brand_result = None
        if model_loader.brand_classifier and model_loader.brand_classifier.model:
            brand_result = model_loader.brand_classifier.predict(request.screenshot_bytes)
        
        # Logo detection
        logo_result = logo_detector.detect(request.screenshot_bytes)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "brand_detection": brand_result,
            "logo_detection": logo_result,
            "processing_time_ms": processing_time
        }
    
    except Exception as e:
        logger.error(f"Error analyzing screenshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    browser_available = page_renderer.browser is not None
    
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
        version=settings.service_version,
        cnn_model_loaded=model_loader.is_loaded(),
        browser_available=browser_available
    )
