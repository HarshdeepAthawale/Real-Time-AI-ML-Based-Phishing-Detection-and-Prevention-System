# Phase 3: NLP Text Analysis Service (Python)

## Objective
Build a transformer-based NLP service using fine-tuned BERT/RoBERTa models for semantic analysis of email/SMS content, AI-generated content detection, and social engineering indicator identification.

## Prerequisites
- Phase 1 & 2 completed
- Python 3.11+ environment
- GPU access (optional, for training; CPU inference is acceptable)
- Access to pre-trained models or training datasets

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              NLP Service (FastAPI)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Text       │  │   Model      │  │   Feature    │ │
│  │ Preprocessor │→ │   Inference  │→ │  Extractor   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   AI Content │  │   Urgency    │  │   Sentiment  │ │
│  │   Detector   │  │   Analyzer   │  │   Analyzer   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
backend/ml-services/nlp-service/
├── src/
│   ├── main.py                    # FastAPI application
│   ├── models/
│   │   ├── phishing_classifier.py # Main phishing detection model
│   │   ├── ai_detector.py         # AI-generated content detector
│   │   └── feature_extractor.py   # Feature extraction utilities
│   ├── preprocessing/
│   │   ├── text_normalizer.py     # Text normalization
│   │   ├── tokenizer.py           # Custom tokenization
│   │   └── email_parser.py        # Email-specific parsing
│   ├── analyzers/
│   │   ├── urgency_analyzer.py    # Urgency detection
│   │   ├── sentiment_analyzer.py  # Sentiment analysis
│   │   └── social_engineering.py  # Social engineering indicators
│   ├── api/
│   │   ├── routes.py              # API route handlers
│   │   └── schemas.py             # Pydantic models
│   └── utils/
│       ├── model_loader.py        # Model loading utilities
│       └── cache.py              # Redis caching
├── models/                        # Pre-trained model files
│   ├── phishing-bert-v1/
│   ├── ai-detector-v1/
│   └── embeddings/
├── training/                      # Training scripts
│   ├── train_phishing_model.py
│   ├── train_ai_detector.py
│   └── data_preparation.py
├── tests/
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Implementation Steps

### 1. Dependencies Setup

**File**: `backend/ml-services/nlp-service/requirements.txt`

```txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# ML Libraries
torch==2.1.0
transformers==4.35.0
sentence-transformers==2.2.2
onnxruntime==1.16.0
scikit-learn==1.3.2

# NLP Utilities
nltk==3.8.1
spacy==3.7.2
email-validator==2.1.0

# Database & Cache
psycopg2-binary==2.9.9
pymongo==4.6.0
redis==5.0.1

# Utilities
python-multipart==0.0.6
aiofiles==23.2.1
python-dotenv==1.0.0
```

### 2. FastAPI Application Setup

**File**: `backend/ml-services/nlp-service/src/main.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from src.api.routes import router
from src.utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)

# Global model loader
model_loader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_loader
    model_loader = ModelLoader()
    await model_loader.load_all_models()
    logger.info("All models loaded successfully")
    yield
    # Shutdown
    await model_loader.unload_all_models()
    logger.info("Models unloaded")

app = FastAPI(
    title="NLP Analysis Service",
    description="Phishing detection using transformer models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": model_loader.is_loaded() if model_loader else False
    }
```

### 3. Model Architecture

#### 3.1 Phishing Detection Model

**File**: `backend/ml-services/nlp-service/src/models/phishing_classifier.py`

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, List
import numpy as np

class PhishingClassifier:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
    
    def predict(self, text: str) -> Dict:
        """Predict phishing probability for given text"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        phishing_prob = probabilities[0][1].item()
        legitimate_prob = probabilities[0][0].item()
        
        return {
            "phishing_probability": phishing_prob,
            "legitimate_probability": legitimate_prob,
            "confidence": abs(phishing_prob - legitimate_prob),
            "prediction": "phishing" if phishing_prob > 0.5 else "legitimate"
        }
    
    def extract_features(self, text: str) -> Dict:
        """Extract semantic embeddings and features"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Use CLS token embedding
            embeddings = outputs.hidden_states[-1][0][0].cpu().numpy()
        
        return {
            "embeddings": embeddings.tolist(),
            "embedding_dim": len(embeddings)
        }
```

#### 3.2 AI-Generated Content Detector

**File**: `backend/ml-services/nlp-service/src/models/ai_detector.py`

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict

class AIGeneratedDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        # Use a model fine-tuned for AI detection (e.g., GPT-2 detector or RoBERTa-based)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
    
    def detect(self, text: str) -> Dict:
        """Detect if text is AI-generated"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        ai_prob = probabilities[0][1].item()
        human_prob = probabilities[0][0].item()
        
        return {
            "ai_generated_probability": ai_prob,
            "human_written_probability": human_prob,
            "is_ai_generated": ai_prob > 0.7,  # Threshold
            "confidence": abs(ai_prob - human_prob)
        }
```

### 4. Text Preprocessing

**File**: `backend/ml-services/nlp-service/src/preprocessing/text_normalizer.py`

```python
import re
import html
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextNormalizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def normalize(self, text: str) -> str:
        """Normalize text for analysis"""
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (but keep placeholder)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        return text.strip()
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
```

**File**: `backend/ml-services/nlp-service/src/preprocessing/email_parser.py`

```python
import email
from email.header import decode_header
from typing import Dict, List, Optional

class EmailParser:
    def parse(self, raw_email: str) -> Dict:
        """Parse email content"""
        msg = email.message_from_string(raw_email)
        
        # Decode headers
        subject = self._decode_header(msg.get('Subject', ''))
        from_addr = self._decode_header(msg.get('From', ''))
        to_addrs = self._decode_header(msg.get('To', ''))
        
        # Extract body
        body_text, body_html = self._extract_body(msg)
        
        # Extract headers
        headers = {}
        for key, value in msg.items():
            headers[key] = self._decode_header(value)
        
        return {
            "subject": subject,
            "from": from_addr,
            "to": to_addrs,
            "body_text": body_text,
            "body_html": body_html,
            "headers": headers
        }
    
    def _decode_header(self, header: str) -> str:
        """Decode email header"""
        decoded_parts = decode_header(header)
        decoded_string = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                decoded_string += part.decode(encoding or 'utf-8', errors='ignore')
            else:
                decoded_string += part
        return decoded_string
    
    def _extract_body(self, msg) -> tuple[str, Optional[str]]:
        """Extract text and HTML body"""
        body_text = ""
        body_html = None
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    body_text = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif content_type == "text/html":
                    body_html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            content_type = msg.get_content_type()
            payload = msg.get_payload(decode=True)
            if payload:
                if content_type == "text/plain":
                    body_text = payload.decode('utf-8', errors='ignore')
                elif content_type == "text/html":
                    body_html = payload.decode('utf-8', errors='ignore')
        
        return body_text, body_html
```

### 5. Feature Analyzers

**File**: `backend/ml-services/nlp-service/src/analyzers/urgency_analyzer.py`

```python
import re
from typing import Dict, List

class UrgencyAnalyzer:
    def __init__(self):
        # Urgency keywords and patterns
        self.urgency_keywords = [
            'urgent', 'immediately', 'asap', 'hurry', 'expired', 'expiring',
            'suspended', 'locked', 'verify', 'confirm', 'update', 'action required',
            'limited time', 'act now', 'click here', 'verify account'
        ]
        
        self.time_patterns = [
            r'\d+\s*(hour|day|minute)s?\s*(ago|left|remaining)',
            r'expir(es|ed|ing)',
            r'deadline',
            r'within\s+\d+\s*(hour|day|minute)s?'
        ]
    
    def analyze(self, text: str) -> Dict:
        """Analyze urgency indicators"""
        text_lower = text.lower()
        
        # Count urgency keywords
        keyword_count = sum(1 for keyword in self.urgency_keywords if keyword in text_lower)
        
        # Check time patterns
        time_pattern_matches = sum(1 for pattern in self.time_patterns if re.search(pattern, text_lower))
        
        # Calculate urgency score (0-100)
        urgency_score = min(100, (keyword_count * 15) + (time_pattern_matches * 20))
        
        # Check for excessive punctuation (urgency indicator)
        exclamation_count = text.count('!')
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        return {
            "urgency_score": urgency_score,
            "keyword_count": keyword_count,
            "time_pattern_matches": time_pattern_matches,
            "exclamation_count": exclamation_count,
            "caps_words_count": caps_words,
            "is_urgent": urgency_score > 50
        }
```

**File**: `backend/ml-services/nlp-service/src/analyzers/sentiment_analyzer.py`

```python
from transformers import pipeline
from typing import Dict

class SentimentAnalyzer:
    def __init__(self):
        # Use a pre-trained sentiment analysis model
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    
    def analyze(self, text: str) -> Dict:
        """Analyze sentiment"""
        result = self.sentiment_pipeline(text[:512])[0]  # Truncate for model limit
        
        return {
            "sentiment": result['label'],
            "sentiment_score": result['score'],
            "is_negative": result['label'] == 'NEGATIVE'
        }
```

### 6. API Endpoints

**File**: `backend/ml-services/nlp-service/src/api/schemas.py`

```python
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict

class TextAnalysisRequest(BaseModel):
    text: str
    include_features: bool = False

class EmailAnalysisRequest(BaseModel):
    raw_email: str
    include_features: bool = False

class AnalysisResponse(BaseModel):
    phishing_probability: float
    legitimate_probability: float
    confidence: float
    prediction: str
    ai_generated_probability: Optional[float] = None
    urgency_score: Optional[float] = None
    sentiment: Optional[str] = None
    features: Optional[Dict] = None
    processing_time_ms: float
```

**File**: `backend/ml-services/nlp-service/src/api/routes.py`

```python
from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import TextAnalysisRequest, EmailAnalysisRequest, AnalysisResponse
from src.models.phishing_classifier import PhishingClassifier
from src.models.ai_detector import AIGeneratedDetector
from src.preprocessing.text_normalizer import TextNormalizer
from src.preprocessing.email_parser import EmailParser
from src.analyzers.urgency_analyzer import UrgencyAnalyzer
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.utils.model_loader import ModelLoader
import time

router = APIRouter()

# Initialize analyzers (singleton pattern)
text_normalizer = TextNormalizer()
email_parser = EmailParser()
urgency_analyzer = UrgencyAnalyzer()
sentiment_analyzer = SentimentAnalyzer()

def get_models():
    """Dependency to get loaded models"""
    return ModelLoader.get_instance()

@router.post("/analyze-text", response_model=AnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    models: ModelLoader = Depends(get_models)
):
    """Analyze text for phishing indicators"""
    start_time = time.time()
    
    try:
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
        
        # Extract features if requested
        features = None
        if request.include_features:
            features = models.phishing_classifier.extract_features(normalized_text)
            features.update({
                "urgency": urgency_result,
                "sentiment": sentiment_result
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            phishing_probability=phishing_result["phishing_probability"],
            legitimate_probability=phishing_result["legitimate_probability"],
            confidence=phishing_result["confidence"],
            prediction=phishing_result["prediction"],
            ai_generated_probability=ai_result["ai_generated_probability"],
            urgency_score=urgency_result["urgency_score"],
            sentiment=sentiment_result["sentiment"],
            features=features,
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-email", response_model=AnalysisResponse)
async def analyze_email(
    request: EmailAnalysisRequest,
    models: ModelLoader = Depends(get_models)
):
    """Analyze email for phishing indicators"""
    start_time = time.time()
    
    try:
        # Parse email
        parsed_email = email_parser.parse(request.raw_email)
        
        # Combine subject and body for analysis
        combined_text = f"{parsed_email['subject']} {parsed_email['body_text']}"
        normalized_text = text_normalizer.normalize(combined_text)
        
        # Run analysis (same as text analysis)
        # ... (similar to analyze_text endpoint)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalysisResponse(
            # ... (same structure as analyze_text)
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-ai-content")
async def detect_ai_content(text: str, models: ModelLoader = Depends(get_models)):
    """Detect if content is AI-generated"""
    result = models.ai_detector.detect(text)
    return result
```

### 7. Model Loading & Caching

**File**: `backend/ml-services/nlp-service/src/utils/model_loader.py`

```python
import os
from src.models.phishing_classifier import PhishingClassifier
from src.models.ai_detector import AIGeneratedDetector

class ModelLoader:
    _instance = None
    _models_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._models_loaded:
            self.phishing_classifier = None
            self.ai_detector = None
            self._models_loaded = False
    
    async def load_all_models(self):
        """Load all models"""
        model_dir = os.getenv("MODEL_DIR", "./models")
        
        # Load phishing classifier
        phishing_model_path = os.path.join(model_dir, "phishing-bert-v1")
        self.phishing_classifier = PhishingClassifier(phishing_model_path)
        
        # Load AI detector
        ai_model_path = os.path.join(model_dir, "ai-detector-v1")
        self.ai_detector = AIGeneratedDetector(ai_model_path)
        
        self._models_loaded = True
    
    async def unload_all_models(self):
        """Unload models to free memory"""
        self.phishing_classifier = None
        self.ai_detector = None
        self._models_loaded = False
    
    def is_loaded(self) -> bool:
        return self._models_loaded
    
    @classmethod
    def get_instance(cls):
        return cls._instance
```

### 8. Docker Configuration

**File**: `backend/ml-services/nlp-service/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

## Training Pipeline (Optional - for initial model training)

**File**: `backend/ml-services/nlp-service/training/train_phishing_model.py`

```python
# Training script using Hugging Face Transformers
# Fine-tune BERT/RoBERTa on phishing email dataset
# Save model to models/phishing-bert-v1/
```

## Testing

**File**: `backend/ml-services/nlp-service/tests/test_api.py`

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_analyze_text():
    response = client.post("/api/v1/analyze-text", json={
        "text": "Urgent! Verify your account immediately or it will be suspended."
    })
    assert response.status_code == 200
    data = response.json()
    assert "phishing_probability" in data
    assert data["phishing_probability"] > 0.5  # Should detect as phishing
```

## Deliverables Checklist

- [ ] FastAPI application created
- [ ] Phishing detection model integrated
- [ ] AI-generated content detector integrated
- [ ] Text preprocessing pipeline implemented
- [ ] Email parser implemented
- [ ] Urgency analyzer implemented
- [ ] Sentiment analyzer implemented
- [ ] API endpoints created and tested
- [ ] Model loading and caching implemented
- [ ] Docker configuration created
- [ ] Unit tests written
- [ ] API documentation generated

## Performance Targets

- Inference latency: <50ms per request (on CPU)
- Throughput: 100+ requests/second
- Model loading time: <30 seconds
- Memory usage: <2GB per worker

## Next Steps

After completing Phase 3:
1. Deploy NLP service to ECS/EKS
2. Test with real email samples
3. Monitor performance metrics
4. Fine-tune models based on feedback
5. Proceed to Phase 4: URL/Domain Analysis Service
