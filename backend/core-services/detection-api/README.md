# Detection API Service

High-performance API service that orchestrates ML services for real-time threat detection with sub-50ms latency, WebSocket support for event streaming, and an ensemble decision engine.

## Features

- **ML Service Orchestration**: Coordinates NLP, URL, and Visual analysis services
- **Ensemble Decision Engine**: Combines multiple ML model outputs with weighted scoring
- **Redis Caching**: Reduces latency for repeated requests
- **WebSocket Support**: Real-time event streaming for threat notifications
- **Rate Limiting**: Protects against abuse
- **Authentication**: API key-based authentication
- **High Performance**: Optimized for sub-50ms cached responses

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Detection API (Node.js/TypeScript)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Request   │  │   ML Service │  │   Decision   │  │
│  │  Router     │→ │ Orchestrator │→ │   Engine     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Cache      │  │   WebSocket   │  │   Event      │  │
│  │   Manager    │  │   Server      │  │   Streamer    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
npm install
```

## Configuration

Environment variables:

- `PORT`: Server port (default: 3001)
- `NODE_ENV`: Environment (development/production)
- `NLP_SERVICE_URL`: NLP service URL (default: http://nlp-service:8000)
- `URL_SERVICE_URL`: URL service URL (default: http://url-service:8000)
- `VISUAL_SERVICE_URL`: Visual service URL (default: http://visual-service:8000)
- `REDIS_HOST`: Redis host (default: redis)
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_PASSWORD`: Redis password (optional)
- `REDIS_URL`: Redis connection URL (default: redis://redis:6379)
- `CORS_ORIGINS`: Comma-separated list of allowed origins (default: *)

## Running

### Development

```bash
npm run dev
```

### Production

```bash
npm run build
npm start
```

### Docker

```bash
docker build -t detection-api .
docker run -p 3001:3001 detection-api
```

## API Endpoints

### Health Check

```
GET /health
```

Returns service health status including cache and WebSocket connection status.

### Detect Email

```
POST /api/v1/detect/email
Headers:
  X-API-Key: <your-api-key>
  X-Organization-Id: <organization-id> (optional)

Body:
{
  "emailContent": "Raw email content...",
  "organizationId": "uuid" (optional),
  "includeFeatures": false (optional)
}
```

### Detect URL

```
POST /api/v1/detect/url
Headers:
  X-API-Key: <your-api-key>
  X-Organization-Id: <organization-id> (optional)

Body:
{
  "url": "https://example.com",
  "legitimateDomain": "example.com" (optional),
  "legitimateUrl": "https://legitimate.com" (optional),
  "organizationId": "uuid" (optional)
}
```

### Detect Text

```
POST /api/v1/detect/text
Headers:
  X-API-Key: <your-api-key>
  X-Organization-Id: <organization-id> (optional)

Body:
{
  "text": "Text content to analyze...",
  "includeFeatures": false (optional),
  "organizationId": "uuid" (optional)
}
```

## WebSocket API

### Connection

```javascript
const socket = io('http://localhost:3001');

socket.on('connect', () => {
  console.log('Connected');
});

// Subscribe to organization events
socket.emit('subscribe', 'organization-id');

// Listen for threat detections
socket.on('threat_detected', (data) => {
  console.log('Threat detected:', data);
});

// Listen for URL analysis events
socket.on('url_analyzed', (data) => {
  console.log('URL analyzed:', data);
});
```

### Events

- `threat_detected`: Emitted when a threat is detected for a subscribed organization
- `url_analyzed`: Emitted when a URL is analyzed
- `subscribed`: Confirmation of subscription
- `unsubscribed`: Confirmation of unsubscription
- `pong`: Response to ping

## Response Format

```json
{
  "isThreat": true,
  "confidence": 0.85,
  "severity": "high",
  "threatType": "email_phishing",
  "scores": {
    "ensemble": 0.85,
    "nlp": 0.80,
    "url": 0.90,
    "visual": 0.00
  },
  "indicators": [
    "high_urgency_language",
    "ai_generated_content"
  ],
  "metadata": {
    "processingTimeMs": 45,
    "timestamp": "2024-01-01T00:00:00.000Z"
  },
  "cached": false
}
```

## Decision Engine

The decision engine uses a weighted ensemble approach:

- **NLP Score**: 40% weight
- **URL Score**: 40% weight
- **Visual Score**: 20% weight

Threat severity levels:
- `critical`: ensemble score >= 0.9
- `high`: ensemble score >= 0.75
- `medium`: ensemble score >= 0.6
- `low`: ensemble score < 0.6

## Caching

Results are cached in Redis:
- Email/Text: 1 hour TTL
- URL: 2 hours TTL

Cache keys are SHA-256 hashes of the input content.

## Performance Targets

- Detection latency: <50ms (cached), <100ms (new)
- Throughput: 1000+ requests/second
- WebSocket connections: 10,000+ concurrent
- Cache hit rate: >80%

## Testing

```bash
npm test
```

## License

ISC
