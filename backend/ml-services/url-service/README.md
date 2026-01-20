# URL/Domain Analysis Service

A FastAPI-based service for phishing detection using Graph Neural Networks (GNNs) for relationship mapping, redirect chain analysis, homoglyph detection, and domain reputation scoring.

## Features

- **URL Parsing & Normalization**: Parse and normalize URLs for consistent analysis
- **Domain Analysis**: Analyze domain characteristics (length, entropy, suspicious patterns)
- **DNS Analysis**: Query DNS records (A, AAAA, MX, TXT, NS, CNAME)
- **WHOIS Analysis**: Analyze domain registration data and age
- **SSL Certificate Analysis**: Validate SSL certificates and check expiry
- **Homoglyph Detection**: Detect visual similarity attacks using Unicode characters
- **Redirect Chain Tracking**: Track and analyze URL redirect chains
- **Graph Neural Network**: Use GNN models for domain relationship analysis
- **Reputation Scoring**: Calculate domain reputation scores based on multiple factors
- **Caching**: Redis-based caching for improved performance

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           URL/Domain Analysis Service                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   URL        │  │   Domain     │  │   Graph      │  │
│  │  Parser      │→ │  Analyzer   │→ │  Constructor │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   GNN        │  │   Redirect   │  │   Reputation │  │
│  │   Model      │  │   Tracker    │  │   Scorer    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
url-service/
├── src/
│   ├── main.py                    # FastAPI application
│   ├── models/
│   │   ├── gnn_classifier.py      # GNN model for domain classification
│   │   └── reputation_scorer.py  # Domain reputation scoring
│   ├── analyzers/
│   │   ├── url_parser.py          # URL parsing and normalization
│   │   ├── domain_analyzer.py     # Domain analysis
│   │   ├── whois_analyzer.py      # WHOIS data analysis
│   │   ├── dns_analyzer.py         # DNS record analysis
│   │   ├── ssl_analyzer.py         # SSL certificate analysis
│   │   └── homoglyph_detector.py   # Homoglyph attack detection
│   ├── graph/
│   │   ├── graph_builder.py       # Build domain relationship graph
│   │   ├── graph_features.py      # Extract graph features
│   │   └── graph_utils.py          # Graph utility functions
│   ├── crawler/
│   │   ├── redirect_tracker.py     # Track redirect chains
│   │   └── link_extractor.py       # Extract links from pages
│   ├── api/
│   │   ├── routes.py               # API route handlers
│   │   └── schemas.py              # Pydantic models
│   └── utils/
│       ├── cache.py                # Redis caching
│       └── validators.py           # URL/domain validators
├── models/
│   └── gnn-domain-classifier-v1/   # Pre-trained model files
├── training/
│   └── train_gnn_model.py          # Training script
├── tests/                          # Test files
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (create `.env` file):
```env
PORT=8001
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
GNN_MODEL_PATH=./models/gnn-domain-classifier-v1/model.pt
CORS_ORIGINS=*
DEBUG=false
```

3. (Optional) Train the GNN model:
```bash
cd training
python train_gnn_model.py --epochs 50 --train-samples 1000
```

## API Endpoints

### Health Check
```http
GET /health
```

### Analyze URL
```http
POST /api/v1/analyze-url
Content-Type: application/json

{
  "url": "https://example.com",
  "legitimate_domain": "example.com"  # Optional, for homoglyph detection
}
```

**Response:**
```json
{
  "url_analysis": {
    "original_url": "https://example.com",
    "normalized_url": "https://example.com",
    "domain": "example",
    "registered_domain": "example.com",
    "tld": "com"
  },
  "domain_analysis": {
    "length": 11,
    "character_entropy": 3.5,
    "suspicious_patterns": []
  },
  "dns_analysis": {
    "a_records": ["93.184.216.34"],
    "has_dns": true
  },
  "whois_analysis": {
    "registered": true,
    "age_days": 1200,
    "is_suspicious": false
  },
  "ssl_analysis": {
    "has_ssl": true,
    "certificate_valid": true
  },
  "redirect_analysis": {
    "redirect_count": 0,
    "is_suspicious": false
  },
  "reputation_score": {
    "reputation_score": 75.5,
    "reputation_level": "good",
    "is_suspicious": false
  },
  "processing_time_ms": 1234.56
}
```

### Analyze Domain
```http
POST /api/v1/analyze-domain
Content-Type: application/json

{
  "domain": "example.com",
  "include_graph": true  # Optional, include GNN analysis
}
```

### Check Redirect Chain
```http
POST /api/v1/check-redirect-chain
Content-Type: application/json

{
  "url": "https://bit.ly/example"
}
```

### Get Domain Reputation
```http
POST /api/v1/domain-reputation
Content-Type: application/json

{
  "domain": "example.com"
}
```

## Usage Examples

### Python
```python
import requests

# Analyze URL
response = requests.post(
    "http://localhost:8001/api/v1/analyze-url",
    json={"url": "https://suspicious-site.com"}
)
result = response.json()
print(f"Reputation Score: {result['reputation_score']['reputation_score']}")
```

### cURL
```bash
curl -X POST http://localhost:8001/api/v1/analyze-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

## Running the Service

### Development
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### Production (Docker)
```bash
docker build -t url-service .
docker run -p 8001:8001 url-service
```

### Docker Compose
The service is included in the main `docker-compose.yml` file.

## Model Training

To train the GNN model:

```bash
cd training
python train_gnn_model.py \
  --epochs 50 \
  --lr 0.001 \
  --batch-size 32 \
  --train-samples 1000 \
  --val-samples 200 \
  --output-dir ./models/gnn-domain-classifier-v1
```

The training script generates synthetic data for demonstration. In production, you would use real domain relationship data from your database.

## Configuration

### Environment Variables

- `PORT`: Service port (default: 8001)
- `LOG_LEVEL`: Logging level (default: INFO)
- `REDIS_URL`: Redis connection URL (optional)
- `GNN_MODEL_PATH`: Path to GNN model file
- `CORS_ORIGINS`: Comma-separated list of allowed origins
- `DEBUG`: Enable debug mode (default: false)

## Dependencies

- **FastAPI**: Web framework
- **PyTorch Geometric**: Graph neural network library
- **NetworkX**: Graph analysis
- **dnspython**: DNS queries
- **python-whois**: WHOIS lookups
- **cryptography**: SSL certificate analysis
- **BeautifulSoup4**: HTML parsing
- **Redis**: Caching

## Testing

Run tests:
```bash
pytest tests/
```

## Performance

- Average response time: 500-2000ms (depending on external API calls)
- Caching reduces response time for repeated queries
- DNS/WHOIS lookups are the main bottleneck

## Limitations

- WHOIS data may be rate-limited by registrars
- DNS queries may fail for some domains
- SSL analysis requires valid HTTPS connections
- GNN model requires training data with domain relationships

## Next Steps

1. Train GNN model on real domain relationship data
2. Integrate with threat intelligence feeds
3. Add more graph features (PageRank, centrality measures)
4. Implement batch processing for multiple URLs
5. Add rate limiting and request queuing

## License

See main project LICENSE file.
