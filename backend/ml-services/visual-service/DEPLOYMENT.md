# Visual Service Deployment Guide

## Overview

This guide covers deployment steps for the Visual Analysis Service, including model training, logo database setup, and production deployment.

## Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- AWS account (if using S3 for screenshot storage)
- GPU access (optional, for model training)

## Step 1: Install Dependencies

```bash
cd backend/ml-services/visual-service
pip install -r requirements.txt
```

## Step 2: Install Playwright Browsers

```bash
playwright install chromium
playwright install-deps chromium
```

## Step 3: Train CNN Model (Optional but Recommended)

### 3.1 Prepare Dataset

Organize your brand dataset:
```
data/
  brand_0/          # e.g., "paypal"
    screenshot1.png
    screenshot2.png
  brand_1/          # e.g., "microsoft"
    screenshot1.png
    screenshot2.png
  unknown/          # Other/unknown brands
    screenshot1.png
```

### 3.2 Run Training

```bash
python training/train_cnn_model.py \
  --data-dir ./data \
  --output-dir ./models/cnn-brand-classifier-v1 \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --num-brands 100
```

### 3.3 Verify Model

After training, verify:
- `models/cnn-brand-classifier-v1/model.pt` exists
- `models/cnn-brand-classifier-v1/brand_mapping.json` exists

## Step 4: Set Up Logo Database

### Option A: Local Storage

1. Create directory:
```bash
mkdir -p logo_templates
```

2. Add logo images:
```bash
logo_templates/
  paypal_logo.png
  microsoft_logo.png
  amazon_logo.png
```

3. Load templates programmatically (see README.md)

### Option B: S3 Storage (Production)

1. Upload logos to S3:
```bash
aws s3 cp logo_templates/ s3://your-bucket/logos/ --recursive
```

2. Configure S3 access in environment variables

## Step 5: Configure Environment

Create `.env` file:
```env
PORT=8000
LOG_LEVEL=INFO
CNN_MODEL_PATH=./models/cnn-brand-classifier-v1/model.pt
S3_BUCKET_NAME=your-bucket-name
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
CORS_ORIGINS=*
DEBUG=false
```

## Step 6: Test Locally

```bash
# Run service
python -m src.main

# Test health endpoint
curl http://localhost:8000/api/v1/health

# Test analysis endpoint
curl -X POST http://localhost:8000/api/v1/analyze-page \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

## Step 7: Docker Deployment

### 7.1 Build Image

```bash
docker build -t visual-service:latest .
```

### 7.2 Run Container

```bash
docker run -d \
  --name visual-service \
  -p 8000:8000 \
  -e PORT=8000 \
  -e CNN_MODEL_PATH=/app/models/cnn-brand-classifier-v1/model.pt \
  -e S3_BUCKET_NAME=your-bucket \
  -v $(pwd)/models:/app/models \
  visual-service:latest
```

### 7.3 Verify Deployment

```bash
# Check logs
docker logs visual-service

# Test health
curl http://localhost:8000/api/v1/health
```

## Step 8: Production Considerations

### Performance

- Screenshot capture: <5 seconds per page
- DOM analysis: <100ms
- CNN inference: <200ms (GPU), <1s (CPU)
- Total analysis: <10 seconds per URL

### Scaling

- Use ECS/Fargate for container orchestration
- Configure auto-scaling based on queue depth
- Use Redis for caching screenshot results
- Consider GPU instances for CNN inference

### Monitoring

- Monitor Playwright browser instances
- Track screenshot capture failures
- Monitor CNN model inference latency
- Set up CloudWatch alarms for service health

## Troubleshooting

### Playwright Browser Issues

If browsers fail to install:
```bash
# Reinstall browsers
playwright install chromium --force

# Check system dependencies
playwright install-deps chromium
```

### Model Loading Issues

If model fails to load:
- Verify `CNN_MODEL_PATH` environment variable
- Check model file exists and is readable
- Service will use untrained model as fallback

### S3 Upload Issues

If S3 uploads fail:
- Verify AWS credentials
- Check bucket permissions
- Service will continue without S3 (screenshots won't be stored)

## Next Steps

After deployment:
1. Train CNN model on production dataset
2. Build comprehensive logo database
3. Configure monitoring and alerts
4. Set up CI/CD pipeline
5. Proceed to Phase 6: Real-time Detection API
