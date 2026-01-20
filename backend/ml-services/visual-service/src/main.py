from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv
from src.api.routes import router
from src.renderer.page_renderer import PageRenderer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global page renderer
page_renderer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global page_renderer
    try:
        page_renderer = PageRenderer()
        await page_renderer.initialize()
        logger.info("Page renderer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize page renderer: {e}", exc_info=True)
        # Continue startup even if renderer fails (will be initialized on first use)
    
    yield
    
    # Shutdown
    if page_renderer:
        await page_renderer.close()
        logger.info("Page renderer closed")

app = FastAPI(
    title="Visual Analysis Service",
    description="Phishing detection using CNN models for visual and structural analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "visual-service",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
