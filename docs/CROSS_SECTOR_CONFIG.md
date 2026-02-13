# Cross-Sector Configuration Profiles

Configuration profiles for deploying the phishing detection system across different organizational types. Each profile tunes detection thresholds, rate limits, and feature flags.

---

## Profile: SMB (Small/Medium Business)

Optimized for cost efficiency with reasonable protection levels.

```env
# Detection thresholds
DETECTION_THRESHOLD=0.65
HIGH_SEVERITY_THRESHOLD=0.85
BLOCK_ON_SEVERITY=critical

# Performance
RATE_LIMIT_PER_MINUTE=50
CACHE_TTL_SECONDS=600
MAX_CONCURRENT_ANALYSES=10

# Features
ENABLE_VISUAL_ANALYSIS=false
ENABLE_SANDBOX_DETONATION=false
ENABLE_THREAT_INTEL_FEEDS=true
ENABLE_AI_CONTENT_DETECTION=true
ENABLE_SMS_DETECTION=true

# ML Services
INFERENCE_DEVICE=cpu
NLP_BATCH_SIZE=8
URL_REDIRECT_MAX_HOPS=5

# Learning Pipeline
AUTO_RETRAIN_ON_DRIFT=false
DRIFT_CHECK_INTERVAL_HOURS=24
FEED_SYNC_INTERVAL_HOURS=12

# Infrastructure
ECS_MIN_TASKS_DETECTION=1
ECS_MAX_TASKS_DETECTION=3
ECS_MIN_TASKS_NLP=1
ECS_MAX_TASKS_NLP=2
REDIS_MAX_MEMORY_MB=256
```

**Notes:**
- Visual analysis disabled to reduce compute costs
- Sandbox disabled; relies on threat intel feeds
- CPU-only inference; suitable for small workloads
- Longer cache TTL to reduce redundant analyses
- Manual retraining (drift alerts, no auto-retrain)

---

## Profile: Enterprise

Balanced performance, accuracy, and cost for mid-to-large organizations.

```env
# Detection thresholds
DETECTION_THRESHOLD=0.55
HIGH_SEVERITY_THRESHOLD=0.80
BLOCK_ON_SEVERITY=high

# Performance
RATE_LIMIT_PER_MINUTE=200
CACHE_TTL_SECONDS=300
MAX_CONCURRENT_ANALYSES=50

# Features
ENABLE_VISUAL_ANALYSIS=true
ENABLE_SANDBOX_DETONATION=true
ENABLE_THREAT_INTEL_FEEDS=true
ENABLE_AI_CONTENT_DETECTION=true
ENABLE_SMS_DETECTION=true

# ML Services
INFERENCE_DEVICE=cuda
NLP_BATCH_SIZE=16
URL_REDIRECT_MAX_HOPS=10

# Learning Pipeline
AUTO_RETRAIN_ON_DRIFT=true
DRIFT_CHECK_INTERVAL_HOURS=6
DRIFT_THRESHOLD=0.12
FEED_SYNC_INTERVAL_HOURS=4

# Infrastructure
ECS_MIN_TASKS_DETECTION=2
ECS_MAX_TASKS_DETECTION=10
ECS_MIN_TASKS_NLP=2
ECS_MAX_TASKS_NLP=6
REDIS_MAX_MEMORY_MB=1024

# Multi-tenant
ENABLE_ORGANIZATION_ISOLATION=true
PER_ORG_RATE_LIMITS=true
```

**Notes:**
- All detection modalities enabled (NLP + URL + Visual)
- GPU inference for lower latency
- Sandbox detonation for suspicious attachments
- Auto-retraining on drift detection
- Organization-level isolation and rate limits
- More aggressive blocking (high severity and above)

---

## Profile: Government / Regulated

Maximum security posture, compliance-focused, audit trails.

```env
# Detection thresholds (aggressive - prefer false positives over missed threats)
DETECTION_THRESHOLD=0.45
HIGH_SEVERITY_THRESHOLD=0.70
BLOCK_ON_SEVERITY=medium

# Performance
RATE_LIMIT_PER_MINUTE=500
CACHE_TTL_SECONDS=120
MAX_CONCURRENT_ANALYSES=100

# Features
ENABLE_VISUAL_ANALYSIS=true
ENABLE_SANDBOX_DETONATION=true
ENABLE_THREAT_INTEL_FEEDS=true
ENABLE_AI_CONTENT_DETECTION=true
ENABLE_SMS_DETECTION=true

# ML Services
INFERENCE_DEVICE=cuda
NLP_BATCH_SIZE=32
URL_REDIRECT_MAX_HOPS=15

# Learning Pipeline
AUTO_RETRAIN_ON_DRIFT=true
DRIFT_CHECK_INTERVAL_HOURS=2
DRIFT_THRESHOLD=0.08
FEED_SYNC_INTERVAL_HOURS=1

# Infrastructure
ECS_MIN_TASKS_DETECTION=4
ECS_MAX_TASKS_DETECTION=20
ECS_MIN_TASKS_NLP=4
ECS_MAX_TASKS_NLP=10
REDIS_MAX_MEMORY_MB=4096

# Compliance & Audit
ENABLE_AUDIT_LOGGING=true
AUDIT_LOG_RETENTION_DAYS=365
ENABLE_DETECTION_ARCHIVAL=true
DETECTION_ARCHIVE_S3_BUCKET=phishing-detection-audit
LOG_PII_REDACTION=true

# Security Hardening
REQUIRE_MTLS=true
API_KEY_ROTATION_DAYS=30
SESSION_TIMEOUT_MINUTES=15
MAX_LOGIN_ATTEMPTS=3

# Data Sovereignty
DATA_RESIDENCY_REGION=us-gov-west-1
CROSS_REGION_REPLICATION=false

# Multi-tenant
ENABLE_ORGANIZATION_ISOLATION=true
PER_ORG_RATE_LIMITS=true
ORG_DATA_ENCRYPTION_AT_REST=true
```

**Notes:**
- Lower detection thresholds (more sensitive; prefers FP over FN)
- Blocks at medium severity and above
- Full audit logging with 1-year retention
- mTLS required for service-to-service communication
- API key auto-rotation every 30 days
- PII redaction in logs
- Data residency constraints (GovCloud)
- Frequent drift checks (every 2 hours) with low threshold
- Hourly threat feed synchronization

---

## Threshold Tuning Guide

| Parameter | Lower Value Effect | Higher Value Effect |
|-----------|-------------------|---------------------|
| `DETECTION_THRESHOLD` | More detections, higher FPR | Fewer detections, lower FPR |
| `HIGH_SEVERITY_THRESHOLD` | More high-severity alerts | Fewer high-severity alerts |
| `DRIFT_THRESHOLD` | More retraining triggers | Fewer retraining triggers |
| `CACHE_TTL_SECONDS` | Fresher results, more compute | Cached results, less compute |
| `URL_REDIRECT_MAX_HOPS` | May miss deep redirects | Catches more redirect chains |

### Recommended Tuning Process

1. Start with the Enterprise profile
2. Run the red-team simulation: `npx ts-node backend/tests/red-team/phishing_simulation.ts`
3. Check TPR and FPR against targets (95% TPR, <2% FPR)
4. Adjust `DETECTION_THRESHOLD` up if FPR is too high, down if TPR is too low
5. Run validation: `python scripts/validate_models.py`
6. Monitor production metrics for 1 week before finalizing thresholds
