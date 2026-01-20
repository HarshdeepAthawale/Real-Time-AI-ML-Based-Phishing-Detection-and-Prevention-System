from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    PageAnalysisRequest,
    PageAnalysisResponse,
    DOMAnalysisRequest,
    DOMAnalysisResponse,
    VisualComparisonRequest,
    VisualComparisonResponse,
    HealthResponse
)
from src.renderer.page_renderer import PageRenderer
from src.analyzers.dom_analyzer import DOMAnalyzer
from src.analyzers.form_analyzer import FormAnalyzer
from src.analyzers.css_analyzer import CSSAnalyzer
from src.analyzers.logo_detector import LogoDetector
from src.models.cnn_classifier import CNNModelLoader
from src.models.visual_similarity import VisualSimilarityMatcher
from src.utils.s3_uploader import S3Uploader
import time
import base64
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
page_renderer = PageRenderer()
dom_analyzer = DOMAnalyzer()
form_analyzer = FormAnalyzer()
css_analyzer = CSSAnalyzer()
logo_detector = LogoDetector()
cnn_model_loader = CNNModelLoader()
similarity_matcher = VisualSimilarityMatcher()
s3_uploader = S3Uploader()

# Load CNN model on startup
cnn_model = None
try:
    cnn_model = cnn_model_loader.load_model()
    logger.info("CNN model loaded successfully")
except Exception as e:
    logger.warning(f"CNN model not loaded: {e}. Brand detection will be limited.")

@router.post("/analyze-page", response_model=PageAnalysisResponse)
async def analyze_page(request: PageAnalysisRequest):
    """Analyze webpage for phishing indicators"""
    start_time = time.time()
    
    try:
        # Render page
        render_result = await page_renderer.render_page(request.url, wait_time=request.wait_time or 3000)
        
        if "error" in render_result:
            raise HTTPException(status_code=500, detail=render_result["error"])
        
        screenshot_bytes = base64.b64decode(render_result["screenshot"])
        
        # Upload screenshot to S3 (optional)
        screenshot_url = None
        try:
            screenshot_url = s3_uploader.upload_screenshot(screenshot_bytes, request.url)
        except Exception as e:
            logger.warning(f"Failed to upload screenshot to S3: {e}")
        
        # DOM analysis
        dom_analysis = dom_analyzer.analyze(render_result["dom"])
        
        # Form analysis
        form_analysis = form_analyzer.analyze(dom_analysis["forms"])
        
        # CSS analysis
        css_analysis = css_analyzer.analyze(render_result["dom"])
        
        # CNN brand impersonation detection
        brand_prediction = None
        if cnn_model:
            try:
                brand_prediction = cnn_model.predict(screenshot_bytes)
            except Exception as e:
                logger.warning(f"Brand prediction failed: {e}")
        
        # Logo detection
        logo_detection_result = None
        try:
            detected_logos = logo_detector.detect_logos(screenshot_bytes)
            brand_colors = logo_detector.detect_brand_colors(screenshot_bytes)
            logo_detection_result = {
                "logos": detected_logos,
                "brand_colors": brand_colors
            }
        except Exception as e:
            logger.warning(f"Logo detection failed: {e}")
        
        # Visual similarity (if legitimate URL provided)
        similarity_analysis = None
        if request.legitimate_url:
            try:
                legit_render = await page_renderer.render_page(request.legitimate_url)
                if "screenshot" in legit_render:
                    legit_screenshot = base64.b64decode(legit_render["screenshot"])
                    similarity_analysis = similarity_matcher.compare_images(
                        screenshot_bytes,
                        legit_screenshot
                    )
            except Exception as e:
                logger.warning(f"Similarity analysis failed: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return PageAnalysisResponse(
            url=request.url,
            dom_analysis=dom_analysis,
            form_analysis=form_analysis,
            css_analysis=css_analysis,
            brand_prediction=brand_prediction,
            similarity_analysis=similarity_analysis,
            logo_detection=logo_detection_result,
            processing_time_ms=round(processing_time, 2),
            screenshot_url=screenshot_url
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Page analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-visual", response_model=VisualComparisonResponse)
async def compare_visual(request: VisualComparisonRequest):
    """Compare two URLs visually"""
    try:
        render1 = await page_renderer.render_page(request.url1)
        render2 = await page_renderer.render_page(request.url2)
        
        if "error" in render1:
            raise HTTPException(status_code=500, detail=f"Failed to render {request.url1}: {render1['error']}")
        if "error" in render2:
            raise HTTPException(status_code=500, detail=f"Failed to render {request.url2}: {render2['error']}")
        
        screenshot1 = base64.b64decode(render1["screenshot"])
        screenshot2 = base64.b64decode(render2["screenshot"])
        
        comparison = similarity_matcher.compare_images(screenshot1, screenshot2)
        
        return VisualComparisonResponse(
            similarity_score=comparison["similarity_score"],
            hamming_distance=comparison["hamming_distance"],
            is_similar=comparison["is_similar"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visual comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-dom", response_model=DOMAnalysisResponse)
async def analyze_dom(request: DOMAnalysisRequest):
    """Analyze DOM structure only"""
    try:
        analysis = dom_analyzer.analyze(request.html_content)
        
        return DOMAnalysisResponse(
            dom_hash=analysis["dom_hash"],
            element_count=analysis["element_count"],
            forms=analysis["forms"],
            links=analysis["links"],
            images=analysis["images"],
            structure=analysis["structure"],
            is_suspicious=analysis["is_suspicious"]
        )
    except Exception as e:
        logger.error(f"DOM analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="visual-service",
        models_loaded=cnn_model is not None
    )
