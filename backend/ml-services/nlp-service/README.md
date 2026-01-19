# NLP Text Analysis Service

A FastAPI-based service for phishing detection using transformer models (BERT/RoBERTa) for semantic analysis of email/SMS content, AI-generated content detection, and social engineering indicator identification.

## Features

- **Phishing Detection**: Uses fine-tuned BERT/RoBERTa models to detect phishing attempts
- **AI Content Detection**: Identifies AI-generated content
- **Urgency Analysis**: Detects urgency indicators and time-sensitive language
- **Sentiment Analysis**: Analyzes sentiment of text content
- **Social Engineering Detection**: Identifies social engineering tactics
- **Email Parsing**: Parses and analyzes email content
- **Feature Extraction**: Extracts linguistic and semantic features

## Architecture

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
nlp-service/
├── src/
│   ├── main.py                    # FastAPI application
│   ├── models/
│   │   ├── phishing_classifier.py # Main phishing detection model
│   │   ├── ai_detector.py         # AI-generated content detector
│   │   └── feature_extractor.py   # Feature extraction utilities
│   ├── preprocessing/
│   │   ├── text_normalizer.py     # Text normalization
│   │   ├── tokenizer.py           # Custom tokenization
│   │   └── email_parser.py         # Email-specific parsing
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
├── training/                      # Training scripts
├── tests/                         # Test files
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data:
```bash
python -m nltk.downloader punkt stopwords
```

3. Set environment variables (create `.env` file):
```env
MODEL_DIR=./models
DEVICE=cpu
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
PORT=8000
CORS_ORIGINS=http://localhost:3000
```

## Running the Service

### Development
```bash
python -m src.main
```

### Production
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### Docker
```bash
docker build -t nlp-service .
docker run -p 8000:8000 nlp-service
```

## API Endpoints

### Health Check
```
GET /health
```

### Analyze Text
```
POST /api/v1/analyze-text
Body: {
    "text": "Your text here",
    "include_features": false
}
```

### Analyze Email
```
POST /api/v1/analyze-email
Body: {
    "raw_email": "Raw email string",
    "include_features": false
}
```

### Detect AI Content
```
POST /api/v1/detect-ai-content?text=Your text here
```

## Response Format

```json
{
    "phishing_probability": 0.85,
    "legitimate_probability": 0.15,
    "confidence": 0.70,
    "prediction": "phishing",
    "ai_generated_probability": 0.30,
    "urgency_score": 75.0,
    "sentiment": "NEGATIVE",
    "social_engineering_score": 65.0,
    "features": {...},
    "processing_time_ms": 45.2
}
```

## Model Training

To train custom models, use the scripts in the `training/` directory:

```bash
python training/train_phishing_model.py
python training/train_ai_detector.py
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## Performance Targets

- Inference latency: <50ms per request (on CPU)
- Throughput: 100+ requests/second
- Model loading time: <30 seconds
- Memory usage: <2GB per worker

## Configuration

The service uses environment variables for configuration:

- `MODEL_DIR`: Directory containing model files (default: `./models`)
- `DEVICE`: Device to run models on (`cpu` or `cuda`) (default: `cpu`)
- `REDIS_URL`: Redis connection URL for caching
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `PORT`: Service port (default: `8000`)
- `CORS_ORIGINS`: Comma-separated list of allowed CORS origins

## Notes

- The service includes fallback methods when models are not available
- Models are loaded at startup and cached in memory
- Redis caching is optional but recommended for production
- The service is designed to work with CPU inference, but GPU support is available

## License

Part of the Real-Time AI/ML-Based Phishing Detection and Prevention System.
