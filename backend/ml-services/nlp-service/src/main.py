"""FastAPI application entry point"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.api.routes import router
from src.models.model_loader import model_loader
from src.config import settings
from src.utils.logger import logger

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])


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

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware — restrict to internal services only
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://detection-api:3001,http://localhost:3001,http://api-gateway:3000,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
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
