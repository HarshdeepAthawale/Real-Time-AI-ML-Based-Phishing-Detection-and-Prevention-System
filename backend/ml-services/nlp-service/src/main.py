"""FastAPI application entry point"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.models.model_loader import model_loader
from src.config import settings
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    try:
        await model_loader.load_all_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load models: {e}. Service will run without models.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down service...")
    await model_loader.unload_all_models()
    logger.info("Service stopped")


app = FastAPI(
    title="NLP Text Analysis Service",
    description="AI/ML-based phishing detection using transformer models",
    version=settings.service_version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["NLP Analysis"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint (alias)"""
    from src.api.routes import health_check
    return await health_check()
