# MEGA COMPLETION PLAN: 100% Production-Ready Phishing Detection System

> **Owner**: Harshdeep
> **Created**: 2026-02-24
> **Goal**: Take the project from ~85% to 100% production-ready
> **Estimated Total Effort**: 3-4 weeks

---

## Table of Contents

1. [Phase A — Security Hardening (Days 1-2)](#phase-a--security-hardening)
2. [Phase B — ML Model Training & Real Intelligence (Days 3-10)](#phase-b--ml-model-training--real-intelligence)
3. [Phase C — Comprehensive Test Suite (Days 11-15)](#phase-c--comprehensive-test-suite)
4. [Phase D — CI/CD Hardening & DevOps (Days 16-17)](#phase-d--cicd-hardening--devops)
5. [Phase E — Monitoring, Logging & Observability (Days 18-19)](#phase-e--monitoring-logging--observability)
6. [Phase F — Database & Infrastructure Polish (Day 20)](#phase-f--database--infrastructure-polish)
7. [Phase G — Documentation & Final Audit (Days 21-22)](#phase-g--documentation--final-audit)
8. [Dependency Graph](#dependency-graph)
9. [Definition of Done](#definition-of-done)

---

## Phase A — Security Hardening

**Priority**: CRITICAL — Nothing ships without this.
**Duration**: Days 1-2

### A1. Remove Hardcoded Credentials

**Files to fix:**
- `backend/.env` — Remove `POSTGRES_PASSWORD=Harshdeep8432`, replace with `POSTGRES_PASSWORD=CHANGE_ME`
- Root `.env` — Ensure no real passwords are committed
- Run `git log --all -p -- '*.env'` to check if passwords leaked into git history
- If leaked: use `git filter-repo` or BFG Repo-Cleaner to scrub history

**Deliverables:**
- [ ] All `.env` files contain only placeholder values
- [ ] `.env.example` files updated with every required variable (see A4)
- [ ] Git history clean of real credentials

### A2. Fix WebSocket URL Protocol

**File:** Root `.env`, line 6

**Change:**
```
# BEFORE
NEXT_PUBLIC_WS_URL=ws://localhost:3000

# AFTER
NEXT_PUBLIC_WS_URL=http://localhost:3000
```

Socket.io auto-upgrades HTTP to WebSocket. Using `ws://` directly breaks the handshake.

### A3. Lock Down CORS on ML Services

**Files:**
- `backend/ml-services/nlp-service/src/main.py` (lines 40-41)
- `backend/ml-services/url-service/src/main.py` (lines 17-21)
- `backend/ml-services/visual-service/src/main.py`

**Change in each:**
```python
# BEFORE
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# AFTER
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://detection-api:3001,http://localhost:3001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type", "Authorization"],
)
```

### A4. Create Comprehensive `.env.example`

**File:** Root `.env.example` (currently only 19 lines — needs ~60+)

Create a master `.env.example` documenting EVERY variable across all services:

```env
# ============================================
# DATABASE
# ============================================
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=phishing_user
POSTGRES_PASSWORD=CHANGE_ME
POSTGRES_DB=phishing_detection
MONGODB_URI=mongodb://localhost:27017/phishing_detection
REDIS_URL=redis://localhost:6379

# ============================================
# API GATEWAY (port 3000)
# ============================================
API_PORT=3000
JWT_SECRET=CHANGE_ME_MIN_32_CHARS
API_KEY=CHANGE_ME
RATE_LIMIT_WINDOW_MS=60000
RATE_LIMIT_MAX_REQUESTS=100

# ============================================
# ML SERVICES
# ============================================
NLP_SERVICE_URL=http://localhost:8000
URL_SERVICE_URL=http://localhost:8001
VISUAL_SERVICE_URL=http://localhost:8002
ALLOWED_ORIGINS=http://detection-api:3001,http://localhost:3001

# ============================================
# THREAT INTELLIGENCE (optional)
# ============================================
MISP_URL=
MISP_API_KEY=
ALIENVAULT_OTX_KEY=
VIRUSTOTAL_API_KEY=
PHISHTANK_API_KEY=

# ============================================
# SANDBOX (optional)
# ============================================
CUCKOO_URL=
ANYRUN_API_KEY=

# ============================================
# AWS (production only)
# ============================================
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET_MODELS=

# ============================================
# FRONTEND
# ============================================
NEXT_PUBLIC_API_URL=http://localhost:3000
NEXT_PUBLIC_WS_URL=http://localhost:3000
```

### A5. Add Rate Limiting to ML Services

**Files:** All three `main.py` in ml-services

Add `slowapi` rate limiter:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/analyze")
@limiter.limit("60/minute")
async def analyze(request: Request, ...):
    ...
```

**Deliverable:** Each ML service has rate limiting configured via environment variable.

### A6. JWT Secret Enforcement

**File:** `backend/core-services/api-gateway/src/index.ts`

Add startup validation:
```typescript
if (!process.env.JWT_SECRET || process.env.JWT_SECRET === 'your-jwt-secret-change-in-production') {
  logger.error('FATAL: JWT_SECRET is not configured. Refusing to start.');
  process.exit(1);
}
```

---

## Phase B — ML Model Training & Real Intelligence

**Priority**: HIGH — Core value proposition of the project.
**Duration**: Days 3-10

### B1. NLP Service — Train Real BERT Phishing Detector

**Directory:** `backend/ml-services/nlp-service/`

**Step 1 — Dataset Acquisition:**
- Download [Phishing Email Dataset from Kaggle](https://www.kaggle.com/datasets) (CEAS, Nazario, or similar)
- Minimum 10,000 samples (5k phishing, 5k legitimate)
- Store in `backend/ml-services/nlp-service/data/training/`

**Step 2 — Training Script:**
Update `scripts/train_phishing_model.py`:
```
- Load dataset (CSV: text, label)
- Tokenize with BERT tokenizer (max_length=256)
- Fine-tune bert-base-uncased for binary classification
- Train: 80/10/10 split, 3-5 epochs, lr=2e-5, batch_size=16
- Evaluate: Precision, Recall, F1, AUC-ROC
- Target: F1 >= 0.95 on test set
- Export: Save model + tokenizer to models/phishing-detector/
```

**Step 3 — Validation:**
- Run `scripts/validate_models.py` — must pass health check
- Test inference via `/analyze` endpoint with known phishing samples

**Deliverables:**
- [ ] Trained BERT model achieving F1 >= 0.95
- [ ] Model saved in `models/phishing-detector/` (config.json, model.safetensors, tokenizer)
- [ ] Training metrics logged and saved to `models/phishing-detector/training_metrics.json`

### B2. NLP Service — Train AI-Generated Content Detector

**Script:** `scripts/train_ai_detector.py`

**Dataset:**
- Mix of human-written and AI-generated phishing emails
- Use GPT-generated text samples + human samples
- Minimum 5,000 samples

**Architecture:**
- Fine-tune `distilbert-base-uncased` for binary classification (AI vs Human)
- Same training pipeline as B1
- Target: F1 >= 0.90

**Deliverables:**
- [ ] Trained AI detector model in `models/ai-detector/`
- [ ] Inference returns `ai_generated_probability` score

### B3. URL Service — Train Real GNN Domain Classifier

**Directory:** `backend/ml-services/url-service/`

**Step 1 — Dataset:**
- PhishTank verified dataset (URLs + labels)
- Alexa Top 1M for benign URLs
- Minimum 50,000 URLs (25k phishing, 25k benign)

**Step 2 — Feature Engineering:**
Update `scripts/create_gnn_model.py` → rename to `scripts/train_gnn_model.py`:
```
Features to extract per URL:
- Domain length, subdomain count, path depth
- Special character ratios (dots, hyphens, digits)
- TLD classification (common vs rare)
- Domain age (via WHOIS lookup, cached)
- URL entropy score
- Character n-gram embeddings
- Graph features: domain→IP→ASN relationships
```

**Step 3 — Model:**
- Graph Neural Network (PyTorch Geometric)
- Node features: URL lexical features
- Edge features: domain-IP-ASN relationships
- 3-layer GCN with 128 hidden units
- Target: AUC-ROC >= 0.97

**Deliverables:**
- [ ] Trained GNN model in `models/gnn-domain-classifier/`
- [ ] Feature extraction pipeline tested with live URLs
- [ ] Model returns risk_score between 0.0 and 1.0

### B4. Visual Service — Train Real CNN Screenshot Classifier

**Directory:** `backend/ml-services/visual-service/`

**Step 1 — Dataset:**
- Screenshot dataset of phishing vs legitimate login pages
- Source: PhishIntention dataset or custom collection using Playwright
- Minimum 10,000 screenshots (5k phishing, 5k legit)
- Resolution: 224x224 (resize pipeline)

**Step 2 — Training:**
Update `scripts/train_cnn_classifier.py`:
```
- Base model: ResNet-50 pretrained on ImageNet
- Replace final FC layer for binary classification
- Fine-tune last 2 residual blocks + new FC
- Augmentation: random crop, horizontal flip, color jitter
- Train: 4 epochs, lr=1e-4, Adam optimizer
- Target: Accuracy >= 0.93
```

**Step 3 — Brand Detection (Optional Enhancement):**
- Add brand logo detection using template matching
- Top 50 most-phished brands (PayPal, Microsoft, Google, etc.)
- Store brand logos in `models/brand-logos/`

**Deliverables:**
- [ ] Trained CNN model in `models/screenshot-classifier/`
- [ ] Inference returns `visual_risk_score` + `matched_brand` (if applicable)
- [ ] Screenshots processed in < 200ms per image

### B5. Learning Pipeline — Local Training Fallback

**File:** `backend/core-services/learning-pipeline/src/index.ts`

**Problem:** Currently requires AWS for model retraining. Need local-only mode.

**Implementation:**
```
- Add LOCAL_TRAINING_MODE=true environment variable
- When enabled, training uses local disk instead of S3
- Trained models saved to shared Docker volume: /models/
- ML services watch /models/ for updated weights (inotify or polling)
- Add /api/learning/retrain POST endpoint that triggers local training
- Add /api/learning/status GET endpoint showing training progress
```

**Deliverables:**
- [ ] Local training works without AWS credentials
- [ ] Model hot-reload on ML services after retraining
- [ ] Training status visible via API

---

## Phase C — Comprehensive Test Suite

**Priority**: HIGH — No production deployment without tests.
**Duration**: Days 11-15

### C1. Backend Unit Tests — Core Services

**Test framework:** Jest + supertest (already in devDependencies)

**For each service, create `tests/` directory with:**

#### API Gateway (`backend/core-services/api-gateway/tests/`)
```
api-gateway.test.ts
├── Authentication middleware tests
│   ├── Valid API key → 200
│   ├── Missing API key → 401
│   ├── Invalid API key → 403
│   └── Expired API key → 403
├── Rate limiting tests
│   ├── Under limit → 200
│   └── Over limit → 429
├── Proxy routing tests
│   ├── /api/detect → detection-api
│   ├── /api/threat-intel → threat-intel
│   └── /api/sandbox → sandbox-service
└── WebSocket proxy tests
    ├── Connection establishment
    └── Event forwarding
```

#### Detection API (`backend/core-services/detection-api/tests/`)
```
detection-api.test.ts
├── URL detection
│   ├── Known phishing URL → high risk score
│   ├── Legitimate URL → low risk score
│   └── Invalid URL format → 400
├── Email detection
│   ├── Phishing email content → threat detected
│   ├── Clean email → no threat
│   └── Empty body → 400
├── Dashboard stats endpoint
├── Recent threats endpoint
└── Caching behavior (Redis)
```

#### Threat Intel (`backend/core-services/threat-intel/tests/`)
```
threat-intel.test.ts
├── IOC management (CRUD)
├── Feed management (CRUD)
├── Feed sync mechanism
├── Intelligence queries
│   ├── Domain lookup
│   ├── Pattern matching
│   └── IOC search
└── External feed integration (mocked)
```

#### Sandbox Service (`backend/core-services/sandbox-service/tests/`)
```
sandbox-service.test.ts
├── File submission
├── URL submission
├── Status polling
├── Results retrieval
└── Disabled adapter behavior (no provider configured)
```

#### Learning Pipeline (`backend/core-services/learning-pipeline/tests/`)
```
learning-pipeline.test.ts
├── Drift detection logic
├── Training job creation
├── Model versioning
└── Local training fallback
```

**Deliverables:**
- [ ] Each core service has >= 80% code coverage
- [ ] All tests pass in CI
- [ ] Tests use mocks for external dependencies (DB, Redis, ML services)

### C2. ML Service Tests (Python)

**Test framework:** pytest + httpx (for async FastAPI testing)

**For each ML service:**

#### NLP Service (`backend/ml-services/nlp-service/tests/`)
```python
test_nlp_service.py
├── test_health_endpoint → 200
├── test_analyze_phishing_text → risk_score > 0.8
├── test_analyze_clean_text → risk_score < 0.3
├── test_analyze_empty_input → 400
├── test_analyze_oversized_input → truncated, still works
├── test_ai_detection → returns ai_probability
└── test_model_loading → models loaded on startup
```

#### URL Service (`backend/ml-services/url-service/tests/`)
```python
test_url_service.py
├── test_health_endpoint → 200
├── test_analyze_phishing_url → high risk
├── test_analyze_legitimate_url → low risk
├── test_analyze_malformed_url → handled gracefully
└── test_feature_extraction → correct feature vector shape
```

#### Visual Service (`backend/ml-services/visual-service/tests/`)
```python
test_visual_service.py
├── test_health_endpoint → 200
├── test_analyze_screenshot → returns risk_score
├── test_analyze_invalid_image → 400
└── test_analyze_oversized_image → resized, still works
```

**Deliverables:**
- [ ] Each ML service has pytest suite
- [ ] Tests run with dummy/small models for speed
- [ ] CI runs `pytest` for each service

### C3. Integration Tests

**Directory:** `backend/tests/integration/`

```
integration/
├── detection-flow.test.ts
│   └── Submit URL → API Gateway → Detection API → ML Services → Response
├── threat-intel-sync.test.ts
│   └── Create feed → Trigger sync → Verify IOCs created
├── websocket-events.test.ts
│   └── Submit detection → WebSocket emits real-time event
├── extension-flow.test.ts
│   └── Extension URL check → API → Block/Allow response
└── docker-compose.integration.yml  (slim compose for integration tests)
```

**Deliverables:**
- [ ] Integration tests run against Docker Compose services
- [ ] `npm run test:integration` script in root package.json
- [ ] Tests pass in CI with service health check gates

### C4. Frontend Tests

**Test framework:** Vitest + React Testing Library (Next.js compatible)

**Directory:** `__tests__/` or colocated `*.test.tsx`

```
Frontend tests:
├── components/
│   ├── api-key-banner.test.tsx
│   ├── threat-chart.test.tsx
│   └── navigation.test.tsx
├── pages/
│   ├── dashboard.test.tsx — renders stats, chart, recent threats
│   ├── detection.test.tsx — form submission, results display
│   ├── intelligence.test.tsx — tabs render, data loads
│   └── settings.test.tsx — form saves preferences
└── hooks/
    └── use-api.test.ts — API call mocking
```

**Deliverables:**
- [ ] Key frontend components have tests
- [ ] `npm run test` in frontend works
- [ ] Tests run in CI

### C5. E2E Tests Enhancement

**File:** `e2e/smoke.spec.ts` (currently exists but minimal)

**Expand with Playwright:**
```
e2e/
├── smoke.spec.ts (existing)
├── detection-flow.spec.ts
│   ├── Navigate to /detection
│   ├── Enter phishing URL
│   ├── Submit
│   └── Verify threat result displayed
├── dashboard.spec.ts
│   ├── Navigate to /
│   ├── Verify stats cards render
│   └── Verify chart renders
└── settings.spec.ts
    ├── Change API key
    └── Verify saved
```

**Deliverables:**
- [ ] 5+ E2E test scenarios covering critical user flows
- [ ] `npx playwright test` passes

---

## Phase D — CI/CD Hardening & DevOps

**Priority**: HIGH
**Duration**: Days 16-17

### D1. Add Security Scanning to CI

**File:** `.github/workflows/security-scan.yml` (NEW)

```yaml
name: Security Scan
on: [push, pull_request]
jobs:
  dependency-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm audit --production
      - run: pip install safety && safety check -r requirements.txt

  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'HIGH,CRITICAL'

  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: github/codeql-action/init@v3
        with:
          languages: javascript, python
      - uses: github/codeql-action/analyze@v3
```

### D2. Enhance Frontend CI

**File:** `.github/workflows/frontend-ci.yml`

**Add:**
- Run frontend tests (`npm test`)
- Build Docker image
- Push to ECR (on main branch merge)
- Deploy to ECS

### D3. Add Test Stage to Backend CI

**File:** `.github/workflows/backend-ci.yml`

**Add:**
- Run unit tests for each service
- Run integration tests against Docker Compose
- Generate coverage reports
- Fail pipeline if coverage < 80%

### D4. Add Pre-commit Hooks

**File:** `.pre-commit-config.yaml` (NEW)

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: detect-private-key
      - id: check-added-large-files
        args: ['--maxkb=5000']
      - id: no-commit-to-branch
        args: ['--branch', 'main']
  - repo: https://github.com/Yelp/detect-secrets
    hooks:
      - id: detect-secrets
```

**Deliverables:**
- [ ] Security scanning runs on every PR
- [ ] Frontend has full CI/CD pipeline
- [ ] Backend CI includes tests with coverage gating
- [ ] Pre-commit hooks prevent credential leaks

---

## Phase E — Monitoring, Logging & Observability

**Priority**: MEDIUM-HIGH
**Duration**: Days 18-19

### E1. Structured Logging Across All Services

**Problem:** Winston logger used inconsistently; some services use `console.log`.

**For each Node.js service**, ensure:
```typescript
import { logger } from '@shared/utils/logger';

// Every log must include:
logger.info('Detection completed', {
  service: 'detection-api',
  requestId: req.id,
  url: sanitizedUrl,
  riskScore: result.score,
  duration: endTime - startTime,
});
```

**For each Python service**, ensure:
```python
import structlog
logger = structlog.get_logger()

logger.info("analysis_complete", service="nlp-service", risk_score=0.85, duration_ms=120)
```

**Deliverables:**
- [ ] All `console.log` replaced with structured logger calls
- [ ] Every log entry includes: service name, request ID, timestamp
- [ ] Python services use `structlog` for JSON logging

### E2. Health Check Standardization

**Every service must expose:**
```
GET /health → { status: "healthy", version: "1.0.0", uptime: 12345 }
GET /health/ready → { ready: true, dependencies: { db: "ok", redis: "ok" } }
```

**Check:** Some services have `/health` but not `/health/ready`. Add readiness probes that verify dependency connectivity.

### E3. Add Prometheus Metrics (Optional but Recommended)

**For Node.js services:** Use `prom-client`
```
Metrics to expose:
- http_request_duration_seconds (histogram)
- http_requests_total (counter, by status code)
- active_websocket_connections (gauge)
- detection_requests_total (counter, by type: url/email/text)
- detection_risk_score (histogram)
- ml_service_latency_seconds (histogram, by service)
```

**For Python services:** Use `prometheus-fastapi-instrumentator`

**Expose:** `GET /metrics` on each service

### E4. Add Docker Compose Monitoring Stack (Optional)

**File:** `docker-compose.monitoring.yml` (NEW)

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3100:3000"
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
```

**Deliverables:**
- [ ] All services have structured JSON logging
- [ ] All services have `/health` and `/health/ready` endpoints
- [ ] Prometheus metrics exposed (stretch goal)

---

## Phase F — Database & Infrastructure Polish

**Priority**: MEDIUM
**Duration**: Day 20

### F1. Fix Migration Numbering Gap

**Problem:** Migrations jump from `001_initial_schema.sql` to `003_seed_api_key.sql`.

**Fix:**
- Rename `003_seed_api_key.sql` → `002_seed_api_key.sql`
- Update any references in init scripts
- Or create a proper `002_indexes_and_constraints.sql` with performance indexes:

```sql
-- 002_indexes_and_constraints.sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_created_at ON detections(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_risk_score ON detections(risk_score);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_iocs_value ON iocs(value);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_iocs_type ON iocs(type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threats_domain ON threats(domain);
```

### F2. Add Database Backup Automation

**File:** `scripts/db-backup.sh` (NEW)

```bash
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER $POSTGRES_DB | gzip > "$BACKUP_DIR/phishing_db_$TIMESTAMP.sql.gz"

# Retain last 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

**For production (AWS):**
- Enable RDS automated backups (already in Terraform, verify retention = 7 days)
- Add point-in-time recovery configuration

### F3. TypeORM Version Alignment

**Problem:** `(dataSource as any)` workarounds due to version conflicts.

**Fix:**
- Pin exact TypeORM version in `backend/shared/package.json`
- Update all service `package.json` to use workspace reference
- Or: Add `"typeorm"` resolution in root `package.json`:
```json
"resolutions": {
  "typeorm": "0.3.28"
}
```

**Deliverables:**
- [ ] Migrations numbered sequentially
- [ ] Backup script exists and is documented
- [ ] TypeORM version consistent across all services

---

## Phase G — Documentation & Final Audit

**Priority**: MEDIUM
**Duration**: Days 21-22

### G1. Architecture Diagram

**File:** `docs/ARCHITECTURE.md` (NEW)

Include:
```
1. High-level system diagram (ASCII or Mermaid)
   - Client (Browser Extension / Web UI)
   - API Gateway (auth, rate limit, routing)
   - Core Services (detection, threat-intel, sandbox, learning)
   - ML Services (NLP, URL, Visual)
   - Data Layer (PostgreSQL, MongoDB, Redis)
   - External (PhishTank, MISP, VirusTotal)

2. Data flow diagrams:
   - URL Detection Flow
   - Email Detection Flow
   - Threat Intel Sync Flow
   - Model Retraining Flow

3. Network diagram:
   - Docker network topology
   - Port mappings
   - Service dependencies
```

### G2. API Reference Update

**File:** `docs/API_REFERENCE.md` (update existing)

Verify every endpoint is documented with:
- Method + URL
- Request headers (API key, content-type)
- Request body (JSON schema)
- Response body (JSON schema with example)
- Error responses (400, 401, 403, 404, 429, 500)
- curl example

### G3. Developer Setup Guide

**File:** `docs/DEVELOPER_SETUP.md` (NEW)

```markdown
## Prerequisites
- Docker & Docker Compose v2
- Node.js 20+
- Python 3.11+
- Git

## Quick Start
1. Clone the repo
2. Copy .env.example → .env and fill in values
3. Run ./scripts/setup-ml-models.sh
4. docker compose up -d
5. Visit http://localhost:3000

## Running Tests
- Backend unit: npm run test (in each service)
- Backend integration: npm run test:integration
- Frontend: npm test
- E2E: npx playwright test
- ML services: pytest

## Common Issues & Troubleshooting
- Port conflicts
- Docker memory limits (ML services need >= 4GB)
- Model download failures
```

### G4. Update PROJECT_COMPLETION_STATUS.md

Update the existing file to reflect actual state — mark ML models as "Trained" once B1-B4 are done, add test coverage numbers, etc.

### G5. Final Security Audit Checklist

Run through `docs/SECURITY_REVIEW.md` and verify every checkbox:
- [ ] All hardcoded credentials removed
- [ ] CORS restricted on all services
- [ ] Rate limiting on all public endpoints
- [ ] Input validation on all endpoints
- [ ] JWT secret enforced
- [ ] No `console.log` of sensitive data
- [ ] Docker images use non-root user
- [ ] Dependencies have no known critical CVEs

---

## Dependency Graph

```
Phase A (Security) ─────────────────────────────────────────┐
    │                                                        │
    v                                                        │
Phase B (ML Training) ──────────┐                            │
    │                            │                            │
    v                            v                            v
Phase C (Tests) ──────> Phase D (CI/CD) ──────> Phase G (Docs & Audit)
    │                            │
    v                            v
Phase E (Monitoring) ──> Phase F (Database)
```

**Critical Path:** A → B → C → D → G

Phases E and F can run in parallel with C/D.

---

## Definition of Done

The project is 100% complete when ALL of these are true:

| # | Criteria | Verification |
|---|----------|-------------|
| 1 | Zero hardcoded credentials in codebase | `grep -r "Harshdeep8432" .` returns nothing |
| 2 | All ML models trained on real data | Each model has `training_metrics.json` with F1 >= 0.90 |
| 3 | Backend test coverage >= 80% | Jest coverage report |
| 4 | ML service test coverage >= 70% | pytest-cov report |
| 5 | Frontend has component tests | `npm test` passes |
| 6 | Integration tests pass | `npm run test:integration` green |
| 7 | E2E tests pass | `npx playwright test` green |
| 8 | Security scan passes CI | No HIGH/CRITICAL in Trivy/CodeQL |
| 9 | All services have structured logging | No `console.log` in production code |
| 10 | All services have health checks | `curl /health` returns 200 on every service |
| 11 | Architecture documented | `docs/ARCHITECTURE.md` exists with diagrams |
| 12 | API fully documented | Every endpoint in `docs/API_REFERENCE.md` |
| 13 | Developer can set up in < 15 minutes | Following `docs/DEVELOPER_SETUP.md` |
| 14 | `docker compose up` starts all services | All health checks pass within 2 minutes |
| 15 | No `(... as any)` TypeORM workarounds | Clean TypeScript compilation |

---

## Quick Reference — Files to Create/Modify

### New Files
| File | Phase |
|------|-------|
| `docs/ARCHITECTURE.md` | G1 |
| `docs/DEVELOPER_SETUP.md` | G3 |
| `.github/workflows/security-scan.yml` | D1 |
| `.pre-commit-config.yaml` | D4 |
| `backend/tests/integration/` | C3 |
| `backend/core-services/*/tests/` | C1 |
| `backend/ml-services/*/tests/` | C2 |
| `scripts/db-backup.sh` | F2 |
| `docker-compose.monitoring.yml` | E4 |
| `monitoring/prometheus.yml` | E4 |

### Modified Files
| File | Phase | Change |
|------|-------|--------|
| `backend/.env` | A1 | Remove real password |
| Root `.env` | A2 | Fix WS URL protocol |
| Root `.env.example` | A4 | Expand to ~60 vars |
| `backend/ml-services/*/src/main.py` | A3, A5 | CORS + rate limiting |
| `backend/core-services/api-gateway/src/index.ts` | A6 | JWT validation |
| `backend/ml-services/nlp-service/scripts/train_*.py` | B1, B2 | Real training |
| `backend/ml-services/url-service/scripts/create_gnn_model.py` | B3 | Real training |
| `backend/ml-services/visual-service/scripts/train_cnn_classifier.py` | B4 | Real training |
| `backend/core-services/learning-pipeline/src/index.ts` | B5 | Local fallback |
| `.github/workflows/backend-ci.yml` | D3 | Add test stage |
| `.github/workflows/frontend-ci.yml` | D2 | Full pipeline |
| `backend/shared/database/migrations/` | F1 | Fix numbering |
| `docs/PROJECT_COMPLETION_STATUS.md` | G4 | Update status |
| `docs/API_REFERENCE.md` | G2 | Full coverage |

---

> **Start with Phase A. Security is non-negotiable.**
> Then Phase B to make the ML models actually intelligent.
> Then prove it all works with Phase C tests.
> Ship it with Phase D CI/CD.
> Polish with E, F, G.
