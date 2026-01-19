from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from dotenv import load_dotenv
from src.api.routes import router
from src.utils.model_loader import ModelLoader
from src.api.schemas import HealthResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model loader
model_loader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_loader
    try:
        model_loader = ModelLoader()
        await model_loader.load_all_models()
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models during startup: {e}", exc_info=True)
        # Continue startup even if models fail to load (they'll use fallback methods)
    
    yield
    
    # Shutdown
    if model_loader:
        await model_loader.unload_all_models()
        logger.info("Models unloaded")

app = FastAPI(
    title="NLP Analysis Service",
    description="Phishing detection using transformer models",
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model_loader
    return HealthResponse(
        status="healthy",
        models_loaded=model_loader.is_loaded() if model_loader else False,
        service="nlp-service"
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "nlp-service",
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
