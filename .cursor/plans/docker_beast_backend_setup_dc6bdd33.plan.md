---
name: Docker Beast Backend Setup
overview: Build a production-ready phishing detection backend with Docker Compose orchestrating all services, real ML models (NLP, URL, Visual), integrated threat intelligence feeds, public phishing datasets, sandbox analysis, and complete data pipeline from detection to storage.
todos: []
---

# Beast Mode: Production Docker Backend with Full ML Pipeline

## Architecture Overview

```mermaid
graph TB
    subgraph frontend [Frontend Layer]
        NextJS[Next.js Dashboard]
        BrowserExt[Browser Extension]
    end
    
    subgraph gateway [API Gateway Layer]
        APIGateway[API Gateway :3000]
    end
    
    subgraph core [Core Services]
        DetectionAPI[Detection API :3001]
        ThreatIntel[Threat Intel :3002]
        ExtensionAPI[Extension API :3003]
        SandboxService[Sandbox Service :3004]
        LearningPipeline[Learning Pipeline]
    end
    
    subgraph ml [ML Services - Python FastAPI]
        NLPService[NLP Service :8001<br/>Transformer Models]
        URLService[URL Service :8002<br/>Domain Analysis + GNN]
        VisualService[Visual Service :8003<br/>Screenshot + CNN]
    end
    
    subgraph data [Data Layer]
        PostgreSQL[(PostgreSQL<br/>Structured Data)]
        MongoDB[(MongoDB<br/>Documents)]
        Redis[(Redis<br/>Cache + Queue)]
    end
    
    subgraph external [External Data Sources]
        MISP[MISP Threat Intel]
        OTX[AlienVault OTX]
        PhishTank[PhishTank API]
        URLhaus[URLhaus]
        AnyRun[Any.Run Sandbox]
        VirusTotal[VirusTotal API]
    end
    
    NextJS --> APIGateway
    BrowserExt --> ExtensionAPI
    APIGateway --> DetectionAPI
    APIGateway --> ThreatIntel
    ExtensionAPI --> DetectionAPI
    
    DetectionAPI --> NLPService
    DetectionAPI --> URLService
    DetectionAPI --> VisualService
    DetectionAPI --> SandboxService
    
    ThreatIntel --> MISP
    ThreatIntel --> OTX
    ThreatIntel --> PhishTank
    ThreatIntel --> URLhaus
    ThreatIntel --> VirusTotal
    
    SandboxService --> AnyRun
    
    DetectionAPI --> PostgreSQL
    DetectionAPI --> Redis
    ThreatIntel --> PostgreSQL
    ThreatIntel --> Redis
    SandboxService --> PostgreSQL
    
    URLService --> MongoDB
    URLService --> Redis
    
    LearningPipeline --> PostgreSQL
    LearningPipeline --> Redis
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Gateway
    participant DetectionAPI
    participant NLP
    participant URL
    participant Visual
    participant ThreatIntel
    participant DB
    participant WebSocket
    
    User->>Frontend: Submit URL for analysis
    Frontend->>Gateway: POST /api/detect/url
    Gateway->>DetectionAPI: Forward request
    
    par Parallel ML Analysis
        DetectionAPI->>NLP: Analyze page text
        DetectionAPI->>URL: Analyze domain/SSL/DNS
        DetectionAPI->>Visual: Screenshot + brand detection
        DetectionAPI->>ThreatIntel: Check IOC matches
    end
    
    NLP-->>DetectionAPI: NLP scores
    URL-->>DetectionAPI: URL risk score
    Visual-->>DetectionAPI: Visual similarity
    ThreatIntel-->>DetectionAPI: Threat matches
    
    DetectionAPI->>DetectionAPI: Ensemble decision
    DetectionAPI->>DB: Save threat + detection
    DetectionAPI->>WebSocket: Broadcast threat event
    DetectionAPI-->>Gateway: Threat result
    Gateway-->>Frontend: JSON response
    WebSocket-->>Frontend: Real-time update
    Frontend-->>User: Display results
```

## Implementation Plan

### Phase 1: Infrastructure Setup (Core Foundation)

**1.1 Environment Configuration**

Create comprehensive environment files:

- [`backend/.env`](backend/.env) - Master environment with all API keys
- Individual service .env files for isolation
- Secrets management for production credentials

Required environment variables:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/phishing_detection
MONGODB_URL=mongodb://mongodb:27017/phishing_detection
REDIS_URL=redis://redis:6379

# External API Keys
MISP_URL=https://your-misp-instance.com
MISP_API_KEY=your_misp_key
OTX_API_KEY=your_otx_key
VIRUSTOTAL_API_KEY=your_vt_key
ANYRUN_API_KEY=your_anyrun_key
PHISHTANK_API_KEY=your_phishtank_key

# Service URLs
NLP_SERVICE_URL=http://nlp-service:8000
URL_SERVICE_URL=http://url-service:8000
VISUAL_SERVICE_URL=http://visual-service:8000
DETECTION_API_URL=http://detection-api:3001
THREAT_INTEL_URL=http://threat-intel:3002

# ML Model Configuration
MODEL_DIR=/models
DEVICE=cpu
```

**1.2 Docker Compose Enhancements**

Update [`backend/docker-compose.yml`](backend/docker-compose.yml):

- Add volume mounts for ML model persistence
- Configure proper health checks for all services
- Add dependency chains to ensure proper startup order
- Set resource limits (CPU/memory) for ML services
- Configure networks for service isolation

Key changes:

```yaml
volumes:
  ml_models:  # Shared volume for ML models
  
services:
  nlp-service:
    volumes:
      - ml_models:/models
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

**1.3 Database Schema Initialization**

Create initialization scripts in [`backend/shared/database/init`](backend/shared/database/init):

- SQL migration scripts for schema creation
- Indexes for performance optimization
- Views for dashboard queries
- Sample data for development/testing

### Phase 2: ML Services - Pre-trained Models (Zero Training Required)

**2.1 NLP Service Enhancement**

Files to modify:

- [`backend/ml-services/nlp-service/src/models/phishing_classifier.py`](backend/ml-services/nlp-service/src/models/phishing_classifier.py)
- [`backend/ml-services/nlp-service/src/models/ai_detector.py`](backend/ml-services/nlp-service/src/models/ai_detector.py)
- [`backend/ml-services/nlp-service/src/api/routes.py`](backend/ml-services/nlp-service/src/api/routes.py)

Implementation:

- Use `distilbert-base-uncased` for phishing detection (lightweight, fast)
- Use `roberta-base` for AI-generated text detection
- Implement model caching with Redis for repeated requests
- Add batch processing for multiple texts
- Create feature extraction pipeline (urgency detection, suspicious patterns)
- Implement confidence calibration
- Add email header parsing and analysis
- Create URL extraction from email bodies

Key code patterns:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class PhishingClassifier:
    def __init__(self, model_name="distilbert-base-uncased", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        self.device = device
        self.model.to(device)
```

**2.2 URL Service - Real Domain Analysis**

Files to modify:

- [`backend/ml-services/url-service/src/analyzers/domain_analyzer.py`](backend/ml-services/url-service/src/analyzers/domain_analyzer.py)
- [`backend/ml-services/url-service/src/analyzers/ssl_analyzer.py`](backend/ml-services/url-service/src/analyzers/ssl_analyzer.py)
- [`backend/ml-services/url-service/src/analyzers/whois_analyzer.py`](backend/ml-services/url-service/src/analyzers/whois_analyzer.py)
- [`backend/ml-services/url-service/src/analyzers/dns_analyzer.py`](backend/ml-services/url-service/src/analyzers/dns_analyzer.py)
- [`backend/ml-services/url-service/src/analyzers/homoglyph_detector.py`](backend/ml-services/url-service/src/analyzers/homoglyph_detector.py)
- [`backend/ml-services/url-service/src/models/reputation_scorer.py`](backend/ml-services/url-service/src/models/reputation_scorer.py)

Real analysis features:

- DNS resolution and MX record checks using `dnspython`
- SSL certificate validation (expiry, issuer, SANs) using `ssl` and `cryptography`
- WHOIS lookup (domain age, registrar, privacy protection) using `python-whois`
- Homoglyph detection (IDN homograph attacks) using Unicode confusables
- Redirect chain analysis
- HTTP response analysis (status codes, headers)
- Reputation scoring based on multiple signals

Implementation details:

```python
import dns.resolver
import ssl
import whois
from datetime import datetime

class DomainAnalyzer:
    async def analyze(self, url: str) -> dict:
        return {
            "dns": await self._check_dns(url),
            "ssl": await self._check_ssl(url),
            "whois": await self._check_whois(url),
            "age_days": self._calculate_domain_age(whois_data),
            "is_suspicious": self._evaluate_suspiciousness(...)
        }
```

**2.3 Visual Service - Screenshot & Brand Analysis**

Files to modify:

- [`backend/ml-services/visual-service/src/renderer/page_renderer.py`](backend/ml-services/visual-service/src/renderer/page_renderer.py)
- [`backend/ml-services/visual-service/src/analyzers/logo_detector.py`](backend/ml-services/visual-service/src/analyzers/logo_detector.py)
- [`backend/ml-services/visual-service/src/analyzers/css_analyzer.py`](backend/ml-services/visual-service/src/analyzers/css_analyzer.py)
- [`backend/ml-services/visual-service/src/analyzers/form_analyzer.py`](backend/ml-services/visual-service/src/analyzers/form_analyzer.py)
- [`backend/ml-services/visual-service/src/models/visual_similarity.py`](backend/ml-services/visual-service/src/models/visual_similarity.py)

Implementation:

- Use Playwright for headless browser rendering
- Screenshot capture with configurable viewport
- Logo detection using template matching (OpenCV)
- Visual similarity comparison using perceptual hashing
- CSS analysis for styling patterns
- Form detection and analysis (input types, submission endpoints)
- DOM structure analysis
- Brand impersonation detection

Key implementation:

```python
from playwright.async_api import async_playwright
import cv2
import imagehash
from PIL import Image

class PageRenderer:
    async def capture_screenshot(self, url: str) -> bytes:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle")
            screenshot = await page.screenshot()
            await browser.close()
            return screenshot
```

### Phase 3: Threat Intelligence Integration

**3.1 Threat Intel Service Core**

Files to create/modify:

- [`backend/core-services/threat-intel/src/services/intelligence.service.ts`](backend/core-services/threat-intel/src/services/intelligence.service.ts)
- [`backend/core-services/threat-intel/src/integrations/misp.client.ts`](backend/core-services/threat-intel/src/integrations/misp.client.ts)
- [`backend/core-services/threat-intel/src/integrations/otx.client.ts`](backend/core-services/threat-intel/src/integrations/otx.client.ts)
- [`backend/core-services/threat-intel/src/integrations/phishtank.client.ts`](backend/core-services/threat-intel/src/integrations/phishtank.client.ts)
- [`backend/core-services/threat-intel/src/integrations/urlhaus.client.ts`](backend/core-services/threat-intel/src/integrations/urlhaus.client.ts)
- [`backend/core-services/threat-intel/src/integrations/virustotal.client.ts`](backend/core-services/threat-intel/src/integrations/virustotal.client.ts)

**3.2 MISP Integration**

Implement MISP (Malware Information Sharing Platform) client:

- Fetch events and attributes via MISP REST API
- Support for indicators: URLs, domains, IPs, file hashes
- Automatic sync on schedule (hourly/daily)
- Store IOCs in PostgreSQL with metadata
- Build Bloom filter for fast lookups
```typescript
export class MISPClient {
  async fetchEvents(days: number = 7): Promise<MISPEvent[]> {
    const response = await axios.post(`${this.baseUrl}/events/restSearch`, {
      published: true,
      publish_timestamp: `${days}d`,
      include_context: true
    }, {
      headers: { 'Authorization': this.apiKey }
    });
    return response.data.response;
  }
}
```


**3.3 AlienVault OTX Integration**

Implement OTX (Open Threat Exchange) client:

- Subscribe to pulse feeds
- Extract IOCs (URLs, domains, IPs, CVEs)
- Parse pulse metadata (threat actor, malware family)
- Store with reputation scores

**3.4 PhishTank Integration**

Implement PhishTank API client:

- Download verified phishing URLs database
- Check URL submissions in real-time
- Cache results in Redis (24h TTL)
- Update database daily

**3.5 URLhaus Integration**

Implement URLhaus (abuse.ch) client:

- Fetch recent malicious URLs
- Get URL analysis reports
- Track malware distribution sites
- Query blacklist for URL reputation

**3.6 VirusTotal Integration**

Implement VirusTotal API v3 client:

- URL scanning and reports
- Domain/IP reputation checks
- File hash lookups
- Aggregate vendor detections
- Rate limiting (4 requests/min for free tier)
```typescript
export class VirusTotalClient {
  async scanUrl(url: string): Promise<VirusTotalReport> {
    // Submit URL for scanning
    const scanResponse = await this.post('/urls', { url });
    const analysisId = scanResponse.data.id;
    
    // Poll for results
    await this.waitForAnalysis(analysisId);
    
    // Get full report
    return await this.getReport(analysisId);
  }
}
```


**3.7 IOC Matching & Bloom Filters**

Implement high-performance IOC matching:

- Build Bloom filter for millions of IOCs
- Fast O(1) lookup for URL/domain checks
- Periodic refresh from threat feeds
- False positive handling with database verification

Files to create:

- [`backend/core-services/threat-intel/src/services/bloom-filter.service.ts`](backend/core-services/threat-intel/src/services/bloom-filter.service.ts)
- [`backend/core-services/threat-intel/src/services/ioc-matcher.service.ts`](backend/core-services/threat-intel/src/services/ioc-matcher.service.ts)

### Phase 4: Sandbox Service Integration

**4.1 Any.Run API Integration**

Files to modify:

- [`backend/core-services/sandbox-service/src/integrations/anyrun.client.ts`](backend/core-services/sandbox-service/src/integrations/anyrun.client.ts)
- [`backend/core-services/sandbox-service/src/services/sandbox-submitter.service.ts`](backend/core-services/sandbox-service/src/services/sandbox-submitter.service.ts)
- [`backend/core-services/sandbox-service/src/services/result-processor.service.ts`](backend/core-services/sandbox-service/src/services/result-processor.service.ts)

Implementation:

- Submit URLs/files for sandbox analysis
- Poll for analysis completion
- Parse behavioral analysis results
- Extract network connections, processes, registry changes
- Detect malware families
- Generate risk scores

**4.2 Cuckoo Sandbox Integration (Optional)**

For self-hosted sandbox:

- Submit samples via Cuckoo REST API
- Configure analysis timeout
- Retrieve JSON reports
- Extract behavioral indicators

**4.3 Queue-based Analysis**

Implement BullMQ job queue:

- Queue URL submissions for sandbox analysis
- Background workers process jobs
- Retry logic for failed submissions
- Status tracking in database
- Webhook notifications on completion
```typescript
export class SandboxQueue {
  async submitUrl(url: string, priority: number = 0): Promise<string> {
    const job = await this.queue.add('analyze-url', {
      url,
      submittedAt: new Date()
    }, {
      priority,
      attempts: 3,
      backoff: { type: 'exponential', delay: 2000 }
    });
    return job.id;
  }
}
```


### Phase 5: Detection Pipeline & Orchestration

**5.1 Enhanced Orchestrator Service**

Files to modify:

- [`backend/core-services/detection-api/src/services/orchestrator.service.ts`](backend/core-services/detection-api/src/services/orchestrator.service.ts)
- [`backend/core-services/detection-api/src/services/decision-engine.service.ts`](backend/core-services/detection-api/src/services/decision-engine.service.ts)

Enhancements:

- Parallel execution of all ML services
- Timeout handling for slow services
- Fallback mechanisms for service failures
- Weighted ensemble scoring
- Threat intelligence IOC matching
- Sandbox submission for suspicious URLs
- Risk score calibration

**5.2 Decision Engine**

Implement sophisticated decision logic:

```typescript
export class DecisionEngineService {
  makeDecision(mlResponse: MLServiceResponse): Threat {
    const weights = {
      nlp: 0.25,
      url: 0.30,
      visual: 0.25,
      threatIntel: 0.20
    };
    
    const scores = {
      nlp: mlResponse.nlp?.confidence || 0,
      url: mlResponse.url?.risk_score || 0,
      visual: mlResponse.visual?.similarity_score || 0,
      threatIntel: mlResponse.ioc_matches ? 1.0 : 0
    };
    
    const finalScore = Object.keys(weights).reduce((sum, key) => 
      sum + weights[key] * scores[key], 0
    );
    
    return {
      isThreat: finalScore > 0.6,
      confidence: finalScore,
      severity: this.calculateSeverity(finalScore),
      threatType: this.determineThreatType(mlResponse),
      indicators: this.extractIndicators(mlResponse)
    };
  }
}
```

**5.3 Cache Strategy**

Implement multi-level caching:

- L1: In-memory LRU cache (hot data)
- L2: Redis cache (warm data, 1-24h TTL)
- L3: Database (cold data, permanent storage)

Cache keys by content hash for deduplication.

### Phase 6: Real-time Event Streaming & WebSockets

**6.1 WebSocket Implementation**

Files to modify:

- [`backend/core-services/detection-api/src/routes/websocket.routes.ts`](backend/core-services/detection-api/src/routes/websocket.routes.ts)
- [`backend/core-services/detection-api/src/services/event-streamer.service.ts`](backend/core-services/detection-api/src/services/event-streamer.service.ts)

Features:

- Real-time threat broadcasting
- Organization-specific rooms
- Authentication with JWT tokens
- Event filtering by severity/type
- Heartbeat for connection monitoring
- Reconnection handling
```typescript
export class EventStreamerService {
  broadcastThreat(orgId: string, threat: Threat) {
    this.io.to(`org:${orgId}`).emit('threat_detected', {
      id: threat.id,
      type: threat.threatType,
      severity: threat.severity,
      confidence: threat.confidence,
      timestamp: new Date()
    });
  }
  
  broadcastStats(orgId: string, stats: DashboardStats) {
    this.io.to(`org:${orgId}`).emit('stats_update', stats);
  }
}
```


**6.2 Event Types**

Define comprehensive event schema:

- `threat_detected` - New threat identified
- `url_analyzed` - URL analysis complete
- `ioc_matched` - Threat intel match found
- `sandbox_complete` - Sandbox analysis finished
- `stats_update` - Dashboard metrics updated
- `feed_sync_complete` - Threat feed synchronized

### Phase 7: Dashboard API & Analytics

**7.1 Dashboard Service**

Files to create:

- [`backend/core-services/detection-api/src/services/dashboard.service.ts`](backend/core-services/detection-api/src/services/dashboard.service.ts)
- [`backend/core-services/detection-api/src/routes/dashboard.routes.ts`](backend/core-services/detection-api/src/routes/dashboard.routes.ts)

Endpoints:

- `GET /api/v1/dashboard/stats` - Overall statistics
- `GET /api/v1/dashboard/threats/recent` - Recent threats
- `GET /api/v1/dashboard/threats/timeline` - Threat timeline
- `GET /api/v1/dashboard/threats/by-type` - Threat distribution
- `GET /api/v1/dashboard/detections/performance` - ML model performance
- `GET /api/v1/dashboard/feeds/status` - Threat feed sync status

**7.2 Analytics Queries**

Implement efficient PostgreSQL queries:

```typescript
async getRecentThreats(orgId: string, limit: number = 50): Promise<Threat[]> {
  return await this.threatRepository.find({
    where: { organization_id: orgId },
    order: { detected_at: 'DESC' },
    take: limit,
    relations: ['detections', 'indicators']
  });
}

async getThreatStats(orgId: string, days: number = 7): Promise<Stats> {
  const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
  
  return {
    total: await this.threatRepository.count({ 
      where: { organization_id: orgId, detected_at: MoreThan(since) }
    }),
    byType: await this.getThreatsByType(orgId, since),
    bySeverity: await this.getThreatsBySeverity(orgId, since),
    timeline: await this.getThreatsTimeline(orgId, since)
  };
}
```

### Phase 8: API Gateway Enhancements

**8.1 Gateway Configuration**

Files to modify:

- [`backend/api-gateway/src/config/gateway.ts`](backend/api-gateway/src/config/gateway.ts)
- [`backend/api-gateway/src/routes/index.ts`](backend/api-gateway/src/routes/index.ts)

Features:

- Route configuration for all services
- Request/response transformation
- API key authentication
- Rate limiting per organization
- Request logging and metrics
- CORS configuration
- Error handling and retries

**8.2 Rate Limiting**

Implement Redis-based rate limiting:

```typescript
export class RateLimiter {
  async checkLimit(apiKey: string): Promise<boolean> {
    const key = `ratelimit:${apiKey}`;
    const requests = await this.redis.incr(key);
    
    if (requests === 1) {
      await this.redis.expire(key, 3600); // 1 hour window
    }
    
    return requests <= 1000; // 1000 requests per hour
  }
}
```

### Phase 9: Testing & Validation

**9.1 Integration Tests**

Create test suites:

- Test ML service endpoints with sample data
- Test detection pipeline end-to-end
- Test threat intelligence IOC matching
- Test WebSocket event broadcasting
- Test API gateway routing

**9.2 Load Testing**

Use k6 or Artillery:

- Test concurrent detection requests
- Measure response times under load
- Identify bottlenecks
- Validate caching effectiveness

**9.3 Sample Data**

Create test fixtures:

- Legitimate emails/URLs
- Known phishing examples
- Malicious URLs from URLhaus
- Brand impersonation samples

### Phase 10: Deployment & Operations

**10.1 Docker Compose Startup**

Complete startup sequence:

```bash
# Navigate to backend directory
cd backend

# Create .env file with all required variables
cp .env.example .env
nano .env  # Add your API keys

# Pull and build all images
docker-compose build

# Start infrastructure services first
docker-compose up -d postgres mongodb redis

# Wait for databases to be ready
docker-compose ps

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
curl http://localhost:3001/health  # Detection API
curl http://localhost:3002/health  # Threat Intel
curl http://localhost:8001/health  # NLP Service
curl http://localhost:8002/health  # URL Service
curl http://localhost:8003/health  # Visual Service
```

**10.2 Database Initialization**

Run migrations:

```bash
# Execute SQL migrations
docker-compose exec postgres psql -U postgres -d phishing_detection -f /migrations/001_initial_schema.sql

# Verify tables
docker-compose exec postgres psql -U postgres -d phishing_detection -c "\dt"
```

**10.3 Model Downloads**

Models auto-download on first startup via HuggingFace Transformers library:

- `distilbert-base-uncased` (~250MB)
- `roberta-base` (~500MB)

Manual download if needed:

```bash
docker-compose exec nlp-service python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('distilbert-base-uncased')
AutoTokenizer.from_pretrained('distilbert-base-uncased')
"
```

**10.4 Threat Feed Initialization**

Trigger initial sync:

```bash
# Start threat intel sync
curl -X POST http://localhost:3002/api/v1/feeds/sync

# Check sync status
curl http://localhost:3002/api/v1/feeds/status
```

**10.5 Monitoring & Logs**

Setup monitoring:

- View aggregated logs: `docker-compose logs -f`
- Individual service logs: `docker-compose logs -f detection-api`
- Resource usage: `docker stats`
- Health checks: `docker-compose ps`

**10.6 Performance Tuning**

Optimize for production:

- Increase worker threads for ML services
- Configure Redis maxmemory policy
- Tune PostgreSQL connection pool
- Enable response compression
- Configure CDN for static assets

## Verification Checklist

After deployment, verify:

- [ ] All services are healthy (green in `docker-compose ps`)
- [ ] Databases are accessible and initialized
- [ ] ML models are loaded (check service logs)
- [ ] API endpoints respond correctly
- [ ] WebSocket connections work
- [ ] Threat feeds are syncing
- [ ] Detection pipeline processes requests
- [ ] Results are saved to database
- [ ] Frontend can connect to backend
- [ ] Real-time updates work via WebSocket

## Testing the System

**Test URL Detection:**

```bash
curl -X POST http://localhost:3001/api/v1/detect/url \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "url": "http://example-phishing-site.com",
    "organizationId": "test-org-id"
  }'
```

**Test Email Detection:**

```bash
curl -X POST http://localhost:3001/api/v1/detect/email \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "emailContent": "Urgent! Your account will be suspended. Click here to verify: http://suspicious-link.com",
    "organizationId": "test-org-id"
  }'
```

**Check Dashboard Stats:**

```bash
curl http://localhost:3001/api/v1/dashboard/stats \
  -H "X-API-Key: your-api-key"
```

## Key Files Reference

### Critical Configuration Files

- [`backend/docker-compose.yml`](backend/docker-compose.yml) - Service orchestration
- [`backend/.env`](backend/.env) - Environment variables
- [`backend/shared/database/models/`](backend/shared/database/models/) - TypeORM models

### ML Service Entry Points

- [`backend/ml-services/nlp-service/src/main.py`](backend/ml-services/nlp-service/src/main.py)
- [`backend/ml-services/url-service/src/main.py`](backend/ml-services/url-service/src/main.py)
- [`backend/ml-services/visual-service/src/main.py`](backend/ml-services/visual-service/src/main.py)

### Core Service Entry Points

- [`backend/core-services/detection-api/src/index.ts`](backend/core-services/detection-api/src/index.ts)
- [`backend/core-services/threat-intel/src/index.ts`](backend/core-services/threat-intel/src/index.ts)
- [`backend/api-gateway/src/index.ts`](backend/api-gateway/src/index.ts)

## Expected Performance

- **URL Detection**: 2-5 seconds (including ML analysis)
- **Email Detection**: 1-3 seconds
- **Threat Intel Lookup**: <100ms (Bloom filter)
- **Sandbox Analysis**: 2-5 minutes (async)
- **WebSocket Latency**: <50ms
- **Throughput**: 100+ requests/sec per service

## Resource Requirements

Minimum system requirements:

- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum (8GB for ML models + services)
- **Storage**: 50GB (models, databases, logs)
- **Network**: Stable internet for API calls

## Security Considerations

- API key authentication on all endpoints
- Rate limiting per organization
- Input validation and sanitization
- SQL injection prevention via ORM
- CORS properly configured
- Secrets in environment variables only
- HTTPS in production
- Regular dependency updates

## Next Steps After Deployment

1. Connect frontend to backend APIs
2. Implement browser extension API integration
3. Set up continuous deployment pipeline
4. Configure monitoring and alerting
5. Implement backup and recovery
6. Scale services based on load
7. Train custom ML models with collected data
8. Fine-tune detection thresholds

---

This plan provides a complete, production-ready implementation with real ML models, comprehensive threat intelligence integration, and scalable architecture. All services run in Docker with proper orchestration, health checks, and data persistence.