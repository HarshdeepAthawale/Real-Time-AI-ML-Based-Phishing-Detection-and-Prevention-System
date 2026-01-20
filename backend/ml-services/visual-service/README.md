# Visual Analysis Service

A FastAPI-based service for phishing detection using CNN models for visual and structural analysis of webpages. This service analyzes webpage screenshots, DOM structures, and visual patterns to detect brand impersonation and phishing attempts.

## Features

- **Page Rendering**: Headless browser rendering using Playwright
- **Screenshot Capture**: Full-page and viewport screenshot capture
- **DOM Analysis**: Structure analysis, form detection, link extraction
- **Form Analysis**: Credential harvesting detection
- **CSS Analysis**: Suspicious CSS pattern detection
- **Logo Detection**: Logo and brand color detection
- **CNN Brand Classification**: Brand impersonation detection using ResNet50
- **Visual Similarity**: Perceptual hashing for visual comparison
- **S3 Integration**: Screenshot storage in AWS S3

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Visual Analysis Service                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Page       │  │   Screenshot │  │   DOM        │  │
│  │   Renderer   │→ │   Capture    │→ │   Parser     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   CNN       │  │   Visual     │  │   Brand      │  │
│  │   Model     │  │   Similarity │  │   Matcher    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
visual-service/
├── src/
│   ├── main.py                    # FastAPI application
│   ├── models/
│   │   ├── cnn_classifier.py      # CNN for brand impersonation
│   │   └── visual_similarity.py   # Visual similarity matching
│   ├── renderer/
│   │   ├── page_renderer.py        # Headless browser rendering
│   │   └── screenshot_capture.py  # Screenshot utilities
│   ├── analyzers/
│   │   ├── dom_analyzer.py         # DOM structure analysis
│   │   ├── form_analyzer.py        # Form field analysis
│   │   ├── css_analyzer.py          # CSS pattern analysis
│   │   └── logo_detector.py        # Logo detection
│   ├── api/
│   │   ├── routes.py               # API route handlers
│   │   └── schemas.py              # Pydantic models
│   └── utils/
│       ├── image_processor.py     # Image processing utilities
│       └── s3_uploader.py          # S3 upload utilities
├── models/
│   └── cnn-brand-classifier-v1/    # Trained model directory
├── training/
│   └── train_cnn_model.py          # Training script
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Playwright browsers:
```bash
playwright install chromium
```

3. Set environment variables (create `.env` file):
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

## Usage

### Running the Service

```bash
# Development
python -m src.main

# Production
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Analyze Page
```bash
POST /api/v1/analyze-page
Content-Type: application/json

{
  "url": "https://example.com",
  "legitimate_url": "https://legitimate-site.com",  # Optional
  "wait_time": 3000  # Optional, milliseconds
}
```

#### Compare Visual
```bash
POST /api/v1/compare-visual
Content-Type: application/json

{
  "url1": "https://example.com",
  "url2": "https://another-site.com"
}
```

#### Analyze DOM
```bash
POST /api/v1/analyze-dom
Content-Type: application/json

{
  "html_content": "<html>...</html>"
}
```

#### Health Check
```bash
GET /api/v1/health
```

## Training the CNN Model

### Dataset Preparation

1. Prepare your dataset with the following structure:
```
data/
  brand_0/
    image1.png
    image2.png
    ...
  brand_1/
    image1.png
    image2.png
    ...
  brand_N/
    ...
  unknown/
    image1.png
    image2.png
    ...
```

Each brand directory should contain screenshots of legitimate brand pages. The `unknown` directory should contain pages that don't match any known brand.

2. Dataset Requirements:
   - Minimum 50 images per brand for reliable training
   - Image formats: PNG, JPG, JPEG
   - Recommended resolution: 1920x1080 or higher
   - Balanced dataset (similar number of images per brand)

### Training Process

Run training:
```bash
python training/train_cnn_model.py \
  --data-dir ./data \
  --output-dir ./models/cnn-brand-classifier-v1 \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --num-brands 100
```

Training will:
- Automatically split data into train/validation sets (80/20)
- Save the best model based on validation accuracy
- Generate `model.pt` and `brand_mapping.json` in the output directory

### Model Deployment

After training, the model will be automatically loaded on service startup if found at `CNN_MODEL_PATH`. The service will gracefully fall back to rule-based detection if the model is not available.

## Logo Database Setup

### Logo Template Storage

Logo templates can be stored in two ways:

1. **Local Storage** (for development):
   - Create directory: `logo_templates/`
   - Add logo images: `logo_templates/brand_name_logo.png`
   - Load templates programmatically using `LogoDetector.add_logo_template()`

2. **S3 Storage** (for production):
   - Upload logos to S3 bucket: `s3://your-bucket/logos/`
   - Reference logos by S3 key in your application
   - Load templates from S3 on service startup

### Logo Template Format

- Recommended format: PNG with transparency
- Recommended size: 200x200 to 500x500 pixels
- File naming: `{brand_name}_logo.png`

### Adding Logo Templates

```python
from src.analyzers.logo_detector import LogoDetector

detector = LogoDetector()

# Load from local file
with open('logo_templates/brand_logo.png', 'rb') as f:
    detector.add_logo_template('brand_id', f.read())

# Or load from S3
import boto3
s3 = boto3.client('s3')
logo_bytes = s3.get_object(Bucket='bucket', Key='logos/brand_logo.png')['Body'].read()
detector.add_logo_template('brand_id', logo_bytes)
```

## Deployment

### Prerequisites

1. **Playwright Browsers**: Must be installed in the container
   - Already included in Dockerfile
   - For manual setup: `playwright install chromium`

2. **Model Files**: 
   - Place trained model at `./models/cnn-brand-classifier-v1/model.pt`
   - Or set `CNN_MODEL_PATH` environment variable

3. **Environment Variables**:
   ```env
   PORT=8000
   LOG_LEVEL=INFO
   CNN_MODEL_PATH=./models/cnn-brand-classifier-v1/model.pt
   S3_BUCKET_NAME=your-bucket-name  # Optional, for screenshot storage
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your-key  # Optional if using IAM role
   AWS_SECRET_ACCESS_KEY=your-secret  # Optional if using IAM role
   CORS_ORIGINS=*
   DEBUG=false
   ```

### Docker Deployment

```bash
# Build image
docker build -t visual-service .

# Run container
docker run -d \
  -p 8000:8000 \
  -e PORT=8000 \
  -e CNN_MODEL_PATH=./models/cnn-brand-classifier-v1/model.pt \
  -e S3_BUCKET_NAME=your-bucket \
  -v $(pwd)/models:/app/models \
  visual-service
```

### Health Check

After deployment, verify the service:
```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "visual-service",
  "models_loaded": true
}
```

### Production Considerations

- **Resource Requirements**: 
  - CPU: 2+ cores recommended
  - RAM: 4GB+ recommended (8GB+ with CNN model)
  - GPU: Optional but recommended for faster CNN inference

- **Scaling**: 
  - Service is stateless and can be horizontally scaled
  - Consider using a load balancer for multiple instances
  - Shared Redis cache recommended for screenshot caching

- **Monitoring**:
  - Monitor `/api/v1/health` endpoint
  - Track processing times via `processing_time_ms` in responses
  - Set up alerts for high error rates

## Docker

Build and run with Docker:

```bash
docker build -t visual-service .
docker run -p 8000:8000 visual-service
```

**Note**: The Dockerfile includes Playwright browser installation. The build process will automatically install Chromium.

## Deployment

### Pre-Deployment Checklist

1. **Install Playwright Browsers**:
   ```bash
   playwright install chromium
   playwright install-deps chromium
   ```

2. **Train CNN Model** (if not using pre-trained):
   - Prepare brand dataset
   - Run training script (see Training section)
   - Verify model.pt exists in `models/cnn-brand-classifier-v1/`

3. **Set Up Logo Database**:
   - Prepare logo templates
   - Upload to S3 or configure local storage
   - Update LogoDetector initialization if needed

4. **Configure Environment Variables**:
   ```env
   PORT=8000
   LOG_LEVEL=INFO
   CNN_MODEL_PATH=./models/cnn-brand-classifier-v1/model.pt
   S3_BUCKET_NAME=your-bucket-name  # Optional, for screenshot storage
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your-access-key  # Optional, if using S3
   AWS_SECRET_ACCESS_KEY=your-secret-key  # Optional, if using S3
   CORS_ORIGINS=*
   DEBUG=false
   ```

5. **Verify Service Health**:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

### Production Deployment

1. **Build Docker Image**:
   ```bash
   docker build -t visual-service:latest .
   ```

2. **Run with Environment Variables**:
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e PORT=8000 \
     -e CNN_MODEL_PATH=/app/models/cnn-brand-classifier-v1/model.pt \
     -e S3_BUCKET_NAME=your-bucket \
     -v $(pwd)/models:/app/models \
     visual-service:latest
   ```

3. **Health Check**:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

### Model Training Status

**Current Status**: Model training infrastructure is complete. The service will use an untrained ResNet50 backbone if no trained model is found. For production use:

- Train the CNN model on a comprehensive brand impersonation dataset
- Place trained model at `models/cnn-brand-classifier-v1/model.pt`
- Update `brand_mapping.json` with your brand names

### Logo Database Status

**Current Status**: Logo detection infrastructure is complete. Logo templates can be added programmatically. For production use:

- Collect logo templates for common brands (PayPal, Microsoft, Amazon, etc.)
- Store in S3 or local directory
- Load templates on service startup or via API

## Performance Considerations

- Screenshot capture: <5 seconds per page
- DOM analysis: <100ms
- CNN inference: <200ms (on GPU), <1s (on CPU)
- Total analysis time: <10 seconds per URL

## Dependencies

- FastAPI: Web framework
- Playwright: Headless browser
- PyTorch: Deep learning framework
- OpenCV: Image processing
- BeautifulSoup4: HTML parsing
- imagehash: Perceptual hashing
- boto3: AWS S3 integration

## License

See project root LICENSE file.
