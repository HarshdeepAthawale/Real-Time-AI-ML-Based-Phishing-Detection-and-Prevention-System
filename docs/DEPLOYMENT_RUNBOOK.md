# Deployment Runbook

## Prerequisites

- Docker and Docker Compose v2+
- AWS account (for production and CI/CD)

---

## 0. GitHub Repository Secrets (CI/CD)

For the backend CI/CD pipeline to build, push images, and deploy, configure these secrets in your GitHub repository (**Settings → Secrets and variables → Actions**):

| Secret | Description | Required For |
|--------|-------------|--------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key for ECR, S3, ECS | Docker push, Terraform, ECS deploy |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key | Docker push, Terraform, ECS deploy |
| `AWS_ACCOUNT_ID` | AWS account ID (12 digits) | ECR login |
| `TF_VAR_DB_PASSWORD` | RDS master password for dev | Terraform plan/apply (dev) |
| `TF_VAR_DB_PASSWORD_PROD` | RDS master password for prod | Terraform apply (prod) |
| `TEST_API_KEY` | API key for smoke and integration tests | Smoke test, integration tests |

Without these secrets, the `build-docker-images`, `terraform-plan`, `deploy-dev`, and `deploy-prod` jobs will fail.

**TEST_API_KEY:** On first `docker compose up`, the database seeds a default test key `testkey_smoke_test_12345` (see `backend/shared/database/init/003_seed_api_key.sql`). Use this for local smoke/integration tests, or set the GitHub secret to the same value. For production, create keys via `backend/shared/scripts/create-initial-setup.ts`.

---
- Node.js 18+ and npm
- Python 3.10+
- AWS account (for production: S3, ECS, CloudWatch)
- Redis 7+
- MongoDB 6+

---

## 1. Environment Variables

### Core Services

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Environment (development/staging/production) | `development` |
| `API_KEY_SECRET` | Secret for API key validation | *required* |
| `JWT_SECRET` | JWT signing secret | *required* |
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/phishing_detection` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `http://localhost:3000` |

### Detection API (port 3001)

| Variable | Description | Default |
|----------|-------------|---------|
| `DETECTION_API_PORT` | Service port | `3001` |
| `NLP_SERVICE_URL` | NLP service base URL | `http://localhost:8000` |
| `URL_SERVICE_URL` | URL service base URL | `http://localhost:8001` |
| `VISUAL_SERVICE_URL` | Visual service base URL | `http://localhost:8002` |
| `CACHE_TTL_SECONDS` | Detection result cache TTL | `300` |
| `RATE_LIMIT_PER_MINUTE` | Rate limit per API key | `100` |

### ML Services

| Variable | Description | Default |
|----------|-------------|---------|
| `PHISHING_MODEL_PATH` | Path to phishing classifier weights | `models/phishing-detector` |
| `AI_DETECTOR_MODEL_PATH` | Path to AI content detector weights | `models/ai-detector` |
| `CNN_MODEL_PATH` | Path to visual CNN weights | `models/cnn-classifier` |
| `INFERENCE_DEVICE` | PyTorch device (cpu/cuda) | `cpu` |
| `MODEL_CACHE_SIZE` | Max models in memory | `3` |

### Threat Intelligence (port 3004)

| Variable | Description | Default |
|----------|-------------|---------|
| `MISP_URL` | MISP server URL | *optional* |
| `MISP_API_KEY` | MISP API key | *optional* |
| `OTX_API_KEY` | AlienVault OTX API key | *optional* |
| `PHISHTANK_API_KEY` | PhishTank API key | *optional* |
| `VIRUSTOTAL_API_KEY` | VirusTotal API key | *optional* |
| `FEED_SYNC_INTERVAL_HOURS` | Auto-sync interval | `6` |

### Learning Pipeline (port 3005)

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_S3_BUCKET` | S3 bucket for training data | *required for prod* |
| `AWS_REGION` | AWS region | `us-east-1` |
| `AUTO_RETRAIN_ON_DRIFT` | Auto-retrain when drift detected | `false` |
| `DRIFT_THRESHOLD` | Drift detection threshold | `0.15` |
| `ECS_CLUSTER_ARN` | ECS cluster for training tasks | *required for prod* |
| `ECS_TASK_DEFINITION` | Training task definition ARN | *required for prod* |

### Extension API (port 3003)

| Variable | Description | Default |
|----------|-------------|---------|
| `GMAIL_CLIENT_ID` | Google OAuth client ID | *optional* |
| `GMAIL_CLIENT_SECRET` | Google OAuth client secret | *optional* |
| `OUTLOOK_CLIENT_ID` | Microsoft OAuth client ID | *optional* |
| `OUTLOOK_CLIENT_SECRET` | Microsoft OAuth client secret | *optional* |

### Sandbox Service (port 3006)

| Variable | Description | Default |
|----------|-------------|---------|
| `CUCKOO_API_URL` | Cuckoo sandbox API URL | *optional* |
| `ANYRUN_API_KEY` | Any.Run API key | *optional* |

---

## 2. Smoke Test

After starting the stack, run the smoke test to validate the detection flow:

```bash
# Ensure services are running (docker compose up)
# Run setup-ml-models.sh before first docker compose up
./scripts/setup-ml-models.sh
docker compose up -d

chmod +x scripts/smoke-test.sh
./scripts/smoke-test.sh http://localhost:3000
# Uses TEST_API_KEY if set; otherwise seeded key testkey_smoke_test_12345
# Or: ./scripts/smoke-test.sh http://localhost:3001 $TEST_API_KEY
```

---

## 3. Local Development Setup

```bash
# Clone and install
git clone <repo-url>
cd Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System

# Start infrastructure
docker compose up -d mongodb redis

# Install Node.js services
cd backend && npm install

# Install Python ML services
cd ml-services/nlp-service && pip install -r requirements.txt
cd ../url-service && pip install -r requirements.txt
cd ../visual-service && pip install -r requirements.txt

# Train models (requires datasets library)
cd nlp-service
python scripts/train_phishing_model.py --epochs 5
python scripts/train_ai_detector.py --epochs 5
cd ../visual-service
python scripts/train_cnn_classifier.py --epochs 10

# Validate models
cd ../nlp-service
python scripts/validate_models.py

# Start ML services
uvicorn src.main:app --host 0.0.0.0 --port 8000 &  # NLP
cd ../url-service && uvicorn src.main:app --host 0.0.0.0 --port 8001 &
cd ../visual-service && uvicorn src.main:app --host 0.0.0.0 --port 8002 &

# Start core services
cd ../../core-services/detection-api && npm run dev &
cd ../extension-api && npm run dev &
cd ../threat-intel && npm run dev &
cd ../learning-pipeline && npm run dev &
```

---

## 3. Docker Deployment

```bash
# Copy env and set POSTGRES_PASSWORD
cp .env.example .env

# Set up ML models (required before first run)
./scripts/setup-ml-models.sh

# Build all services
docker compose build

# Start everything (minimal: postgres, redis, mongodb, api-gateway, detection-api, ML services)
docker compose up -d

# Full stack (threat-intel, extension-api, sandbox-service, learning-pipeline):
# docker compose --profile full up -d

# Check health (ports: 3000=api-gateway, 3001=detection-api, 3002=threat-intel, 8000–8002=ML)
curl http://localhost:3000/health
curl http://localhost:3001/health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

---

## 4. Production Deployment (AWS)

### 4.0 AWS Setup via CLI (One-Time)

For full AWS setup via CLI (bootstrap, ECR, Terraform, secrets):

```bash
export TF_VAR_db_password="your-secure-password"
./scripts/aws-setup.sh all
```

See [docs/AWS_CLI_SETUP.md](AWS_CLI_SETUP.md) for step-by-step and environment variables.

### 4.1 Infrastructure (Terraform)

```bash
cd backend/infrastructure/terraform
terraform init
terraform plan -var-file=environments/prod.tfvars
terraform apply -var-file=environments/prod.tfvars
```

### 4.1.1 Production Checklist

Before production deploy, ensure:

- [ ] `certificate_arn` is set in `environments/prod.tfvars` (ACM certificate for HTTPS)
- [ ] `TF_VAR_db_password_PROD` GitHub secret is configured
- [ ] ML models are built/uploaded (`./scripts/setup-ml-models.sh` for local; S3 sync for ECS)
- [ ] Threat intel API keys (MISP, OTX, PhishTank, etc.) are set if feed sync is required
- [ ] Sandbox keys (ANYRUN_API_KEY or CUCKOO_*) are set if sandbox analysis is required

### 4.2 ECS Service Deployment

Each microservice runs as an ECS Fargate task:

| Service | CPU | Memory | Min Tasks | Max Tasks |
|---------|-----|--------|-----------|-----------|
| Detection API | 512 | 1024 MB | 2 | 10 |
| NLP Service | 1024 | 2048 MB | 2 | 6 |
| URL Service | 512 | 1024 MB | 2 | 6 |
| Visual Service | 1024 | 2048 MB | 1 | 4 |
| Threat Intel | 256 | 512 MB | 1 | 2 |
| Extension API | 256 | 512 MB | 2 | 8 |
| Learning Pipeline | 512 | 1024 MB | 1 | 1 |

### 4.3 Model Deployment

Models are stored in S3 and loaded at container startup:

```bash
# Upload trained models to S3
aws s3 sync models/phishing-detector s3://$BUCKET/models/phishing-detector/
aws s3 sync models/ai-detector s3://$BUCKET/models/ai-detector/
aws s3 sync models/cnn-classifier s3://$BUCKET/models/cnn-classifier/
```

### 4.4 Secrets Management

Store secrets in AWS Secrets Manager or SSM Parameter Store:

```bash
aws ssm put-parameter --name "/phishing-detection/prod/API_KEY_SECRET" --value "..." --type SecureString
aws ssm put-parameter --name "/phishing-detection/prod/JWT_SECRET" --value "..." --type SecureString
aws ssm put-parameter --name "/phishing-detection/prod/MONGODB_URI" --value "..." --type SecureString
```

---

## 5. Scaling Guidelines

### Auto-scaling Triggers

| Metric | Scale Up | Scale Down |
|--------|----------|------------|
| CPU Utilization | > 70% for 2 min | < 30% for 10 min |
| Request Count | > 500 req/min | < 100 req/min |
| p95 Latency | > 200ms for 5 min | - |

### Redis Scaling

- Use Redis Cluster for > 10K concurrent connections
- Set `maxmemory-policy allkeys-lru` for cache eviction
- Monitor cache hit rate; target > 80%

### MongoDB Scaling

- Replica set with 3 nodes for HA
- Shard by `organizationId` for multi-tenant
- Index on `detectionResults.timestamp` and `iocs.value`

---

## 6. Health Checks

Core services expose `/health`:

| Port | Service |
|------|---------|
| 3000 | API Gateway |
| 3001 | Detection API |
| 3002 | Threat Intel |
| 3003 | Extension API |
| 3004 | Sandbox Service |
| 8000 | NLP Service |
| 8001 | URL Service |
| 8002 | Visual Service |

```bash
# Quick health check for core services
for port in 3000 3001 8000 8001 8002; do
  echo -n "Port $port: "
  curl -s http://localhost:$port/health | jq -r '.status // .detail // "error"'
done
```

---

## 7. Rollback Procedure

```bash
# ECS: Roll back to previous task definition
aws ecs update-service \
  --cluster phishing-detection \
  --service detection-api \
  --task-definition detection-api:<previous-revision>

# Docker Compose: Roll back to previous image
docker compose pull
docker compose up -d --no-deps <service-name>

# Model rollback: Restore previous model from S3
aws s3 sync s3://$BUCKET/models/phishing-detector/v<previous>/ models/phishing-detector/
```

---

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| High latency (> 200ms) | Model cold start or no cache | Check Redis connectivity; warm up models |
| 401 on all requests | API key misconfiguration | Verify `API_KEY_SECRET` env var |
| NLP returns 0.5 for all | Model not loaded | Check `PHISHING_MODEL_PATH`; verify model files exist |
| Drift alerts firing | Data distribution shift | Review recent samples; consider retraining |
| Extension not blocking | Manifest permissions | Verify `webRequestBlocking` permission in manifest |
| Feed sync failing | API key expired | Rotate MISP/OTX/PhishTank keys |
