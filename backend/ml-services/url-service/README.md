# URL/Domain Analysis Service

Graph Neural Network-based URL and domain threat analysis service.

## Features

- **URL Parsing & Normalization**: Extract and normalize URL components
- **Domain Analysis**: Analyze domain characteristics and patterns
- **DNS Analysis**: Check DNS records and configuration
- **WHOIS Analysis**: Domain registration and age information
- **SSL Certificate Validation**: Verify SSL certificates
- **Homoglyph Detection**: Detect lookalike domain attacks
- **Redirect Tracking**: Follow HTTP redirect chains
- **GNN Classification**: Graph-based domain classification
- **Reputation Scoring**: Multi-factor domain reputation

## API Endpoints

### POST `/api/v1/analyze-url`
Complete URL analysis with all checks.

### POST `/api/v1/analyze-domain`
Domain-only analysis using GNN model.

### POST `/api/v1/check-redirect-chain`
Track HTTP redirect chains.

### POST `/api/v1/detect-homoglyph`
Check for homoglyph attacks.

### GET `/health`
Health check endpoint.

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

## Docker

```bash
docker build -t url-service .
docker run -p 8001:8001 url-service
```

## Model Placement

Place your GNN model in:
```
models/
└── gnn-domain-classifier/
    ├── model.pth
    └── config.json
```

## Performance

- **Analysis Latency**: <200ms per URL
- **Caching**: 2 hour TTL for results
- **Timeout Settings**: Configurable DNS/WHOIS timeouts
