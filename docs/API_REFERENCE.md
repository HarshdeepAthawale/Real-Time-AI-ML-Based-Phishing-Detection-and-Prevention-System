# API Reference

## Base URLs

| Service | Default Port | Base URL |
|---------|-------------|----------|
| API Gateway | 3000 | `http://localhost:3000/api/v1` |
| Detection API | 3001 | `http://localhost:3001/api/v1` |
| Threat Intel | 3002 | `http://localhost:3002/api/v1` |
| Extension API | 3003 | `http://localhost:3003/api/v1` |
| Sandbox Service | 3004 | `http://localhost:3004/api/v1` |
| Learning Pipeline | 3005 | `http://localhost:3005/api/v1` |
| NLP Service | 8000 | `http://localhost:8000` |
| URL Service | 8001 | `http://localhost:8001` |
| Visual Service | 8002 | `http://localhost:8002` |

## Authentication

All API requests require an API key in the `X-API-Key` header:

```
X-API-Key: your-api-key-here
```

---

## Detection API

### POST /api/v1/detect/email

Analyze an email for phishing indicators.

**Request:**
```json
{
  "emailContent": "Subject: Urgent...\nFrom: sender@example.com\n\nEmail body...",
  "organizationId": "uuid-optional",
  "includeFeatures": false
}
```

**Response:**
```json
{
  "isThreat": true,
  "confidence": 0.92,
  "severity": "high",
  "threatType": "email_phishing",
  "scores": {
    "ensemble": 0.85,
    "nlp": 0.92,
    "url": 0.78,
    "visual": 0.0
  },
  "indicators": ["high_urgency_language", "ai_generated_content"],
  "metadata": {
    "processingTimeMs": 45,
    "timestamp": "2024-12-15T10:30:00Z"
  }
}
```

### POST /api/v1/detect/url

Analyze a URL for phishing indicators.

**Request:**
```json
{
  "url": "https://suspicious-domain.xyz/login",
  "legitimateDomain": "example.com",
  "organizationId": "uuid-optional"
}
```

**Response:**
```json
{
  "isThreat": true,
  "confidence": 0.88,
  "severity": "critical",
  "threatType": "url_spoofing",
  "scores": {
    "ensemble": 0.91,
    "nlp": 0.0,
    "url": 0.95,
    "visual": 0.78
  },
  "indicators": ["homoglyph_attack", "url_obfuscation", "obfuscation_suspicious_tld"],
  "metadata": {
    "processingTimeMs": 120,
    "timestamp": "2024-12-15T10:30:00Z"
  }
}
```

### POST /api/v1/detect/text

Analyze text content for phishing indicators.

**Request:**
```json
{
  "text": "Your account has been compromised...",
  "includeFeatures": true,
  "organizationId": "uuid-optional"
}
```

### POST /api/v1/detect/sms

Analyze SMS message for phishing indicators.

**Request:**
```json
{
  "message": "USPS: Your package is held at customs. Pay $3.99: http://usps-customs.xyz",
  "sender": "+1234567890",
  "organizationId": "uuid-optional"
}
```

**Response:** Same format as email detection, with additional fields:
```json
{
  "isThreat": true,
  "inputType": "sms",
  "sender": "+1234567890",
  "urlsFound": ["http://usps-customs.xyz"]
}
```

### GET /api/v1/metrics/latency

Get latency statistics per endpoint.

**Response:**
```json
{
  "service": "detection-api",
  "timestamp": "2024-12-15T10:30:00Z",
  "endpoints": {
    "/api/v1/detect/email": {
      "count": 150,
      "p50": 32.5,
      "p95": 78.2,
      "p99": 120.1,
      "mean": 41.3
    }
  }
}
```

---

## Threat Intelligence API

### GET /api/v1/intelligence/domains

Get malicious domains list.

**Query Parameters:**
- `limit` (number) - Max results (default: 50)
- `offset` (number) - Pagination offset (default: 0)

**Response:** Array of `{ domain, reputation, reports, firstSeen?, lastSeen? }`

### GET /api/v1/intelligence/patterns

Get threat patterns aggregated from IOCs.

**Query Parameters:**
- `limit` (number) - Max results (default: 20)

**Response:** Array of `{ pattern, incidents, severity? }`

### GET /api/v1/intelligence/iocs

Get IOCs list.

**Query Parameters:**
- `limit`, `offset`, `type` (optional IOC type filter)

**Response:** Array of `{ value, type, sources, firstSeen?, lastSeen?, severity? }`

### GET /api/v1/intelligence/summary

Get threat intelligence summary statistics.

**Response:** `{ knownThreats, feedIntegrations, zeroDayDetection, lastUpdated }`

---

### GET /api/v1/ioc/search (IOC Search)

List IOCs with pagination.

**Query Parameters:**
- `page` (number) - Page number (default: 1)
- `limit` (number) - Items per page (default: 50)
- `type` (string) - Filter by IOC type (url, domain, ip, email, hash)
- `severity` (string) - Filter by severity

### POST /api/v1/iocs/check

Check if an indicator is known malicious.

**Request:**
```json
{
  "type": "url",
  "value": "http://suspicious-site.xyz"
}
```

### GET /api/v1/feeds

List configured threat intelligence feeds.

### POST /api/v1/feeds/sync

Trigger manual feed synchronization.

---

## Sandbox API

### GET /api/v1/sandbox/status

Check if sandbox analysis is enabled.

**Response:** `{ sandboxEnabled, provider, message? }`

### POST /api/v1/sandbox/analyze/file

Submit a file for sandbox analysis (multipart/form-data, field: `file`).

### POST /api/v1/sandbox/analyze/url

Submit a URL for sandbox analysis. Body: `{ url: string }`.

### GET /api/v1/sandbox/analysis/:id

Get analysis status and results.

### GET /api/v1/sandbox/analyses

List analyses with pagination. Query: `page`, `limit`.

---

## Extension API

### POST /api/v1/extension/check-url

Check URL from browser extension.

**Headers:**
- `X-API-Key` - API key
- `X-Extension-Id` - Browser extension ID

**Request:**
```json
{
  "url": "https://example.com",
  "privacyMode": false
}
```

### POST /api/v1/extension/scan-email

Scan email from browser extension (Gmail/Outlook).

### POST /api/v1/extension/report

Report a threat from the extension.

---

## ML Services (Internal)

### NLP Service (port 8000)

- `POST /api/v1/analyze-text` - Analyze text for phishing
- `POST /api/v1/analyze-email` - Analyze email for phishing
- `POST /api/v1/detect-ai-content` - Detect AI-generated content
- `GET /api/v1/health` - Health check
- `GET /api/v1/models/info` - Model status

### URL Service (port 8001)

- `POST /api/v1/analyze-url` - Full URL analysis
- `POST /api/v1/analyze-domain` - Domain-level analysis
- `POST /api/v1/check-redirect-chain` - Check redirect hops
- `POST /api/v1/detect-homoglyph` - Detect typosquatting

### Visual Service (port 8002)

- `POST /api/v1/analyze-page` - Visual page analysis
- `POST /api/v1/analyze-image` - Image analysis
- `POST /api/v1/compare` - Visual similarity comparison

---

## WebSocket Events

Connect to `ws://localhost:3002` (Socket.IO).

### Events Emitted

| Event | Description |
|-------|-------------|
| `threat_detected` | New threat detected |
| `url_analyzed` | URL analysis completed |
| `feed_synced` | Threat feed synchronized |

### Subscribing

```javascript
const socket = io('http://localhost:3002');
socket.on('threat_detected', (data) => {
  console.log('Threat:', data);
});
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": "Validation failed",
  "details": [
    {
      "code": "invalid_type",
      "path": ["url"],
      "message": "Invalid URL format"
    }
  ]
}
```

| Status Code | Meaning |
|------------|---------|
| 400 | Bad Request / Validation Error |
| 401 | Unauthorized (missing/invalid API key) |
| 429 | Rate Limited |
| 500 | Internal Server Error |

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| Detection API | 100 req/min |
| Extension API | 200 req/min |
| Threat Intel | 50 req/min |
