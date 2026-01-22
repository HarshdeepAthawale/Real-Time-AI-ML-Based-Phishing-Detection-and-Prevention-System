# NLP Text Analysis Service

AI/ML-based phishing detection using transformer models (BERT/RoBERTa).

## Features

- **Phishing Detection**: Fine-tuned transformer models for semantic analysis
- **AI Content Detection**: Identify AI-generated phishing attempts
- **Urgency Analysis**: Detect time-pressure tactics
- **Sentiment Analysis**: Identify fear-based social engineering
- **Social Engineering Detection**: Recognize manipulation tactics
- **Email Parsing**: Extract and analyze email headers and body

## API Endpoints

### POST `/api/v1/analyze-text`
Analyze plain text for phishing indicators.

**Request**:
```json
{
  "text": "Urgent! Your account will be suspended...",
  "include_features": false
}
```

**Response**:
```json
{
  "phishing_probability": 0.87,
  "legitimate_probability": 0.13,
  "confidence": 0.74,
  "prediction": "phishing",
  "urgency_score": 75,
  "sentiment": "NEGATIVE",
  "social_engineering_score": 68,
  "processing_time_ms": 45.2,
  "cached": false
}
```

### POST `/api/v1/analyze-email`
Analyze email content for phishing.

**Request**:
```json
{
  "subject": "Verify your account",
  "body": "Click here to verify...",
  "sender": "noreply@suspicious.com",
  "include_features": false
}
```

### POST `/api/v1/detect-ai-content`
Check if content is AI-generated.

### GET `/health`
Health check endpoint.

### GET `/models/info`
Get information about loaded models.

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
# Set environment variables
export MODEL_DIR=/app/models
export REDIS_URL=redis://localhost:6379

# Run service
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

```bash
docker build -t nlp-service .
docker run -p 8000:8000 \
  -e MODEL_DIR=/app/models \
  -e REDIS_URL=redis://redis:6379 \
  nlp-service
```

## Model Placement

Place your trained models in:
```
models/
├── phishing-detector/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── vocab.txt
└── ai-detector/
    ├── pytorch_model.bin
    └── config.json
```

## Configuration

Environment variables:
- `MODEL_DIR`: Directory containing model files (default: `/app/models`)
- `REDIS_URL`: Redis connection URL
- `MONGODB_URL`: MongoDB connection URL
- `INFERENCE_DEVICE`: `cpu` or `cuda` (default: `cpu`)
- `LOG_LEVEL`: Logging level (default: `info`)

## Performance

- **Inference Latency**: <50ms per request (CPU)
- **Throughput**: 100+ requests/second
- **Cache Hit Rate**: >70% for repeated content
