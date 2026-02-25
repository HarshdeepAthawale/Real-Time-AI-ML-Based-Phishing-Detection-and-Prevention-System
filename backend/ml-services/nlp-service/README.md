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
  "ai_generated_probability": 0.92,
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

When run from the `nlp-service` directory, `MODEL_DIR` defaults to `./models` (the service's `models/` folder), so no extra env is needed if your trained model is in `models/phishing-detector/`.

```bash
# Optional: set if using a different model location
# export MODEL_DIR=/path/to/models
export REDIS_URL=redis://localhost:6379

# Run service (from backend/ml-services/nlp-service)
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

Place your trained models in `models/` (or set `MODEL_DIR` to that directory). Example layout:
```
models/
├── phishing-detector/
│   ├── config.json
│   ├── model.safetensors   # or pytorch_model.bin
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── ai-detector/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

To install the Colab-trained **ai-detector** (second model) from a local folder or zip:
```bash
./scripts/download-trained-models.sh --from-dir /path/to/containing/folder
```

## Configuration

Environment variables:
- `MODEL_DIR`: Directory containing model files (default: local `models/` if present, else `/app/models`)
- `REDIS_URL`: Redis connection URL
- `MONGODB_URL`: MongoDB connection URL
- `INFERENCE_DEVICE`: `cpu` or `cuda` (default: `cpu`)
- `LOG_LEVEL`: Logging level (default: `info`)

## Performance

- **Inference Latency**: <50ms per request (CPU)
- **Throughput**: 100+ requests/second
- **Cache Hit Rate**: >70% for repeated content
