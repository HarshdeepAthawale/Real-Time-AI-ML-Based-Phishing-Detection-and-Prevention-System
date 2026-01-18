# Phase 1: Core Infrastructure & Architecture Setup

## Objective
Establish the foundational infrastructure, project structure, and development environment for the entire backend system.

## Prerequisites
- AWS Account with appropriate permissions
- Docker Desktop installed
- Terraform CLI installed
- Node.js 20+ and Python 3.11+ installed
- GitHub account for CI/CD

## Implementation Steps

### 1. Project Structure Setup

Create the following directory structure:

```
backend/
├── api-gateway/              # Node.js/TypeScript API Gateway
│   ├── src/
│   │   ├── routes/
│   │   ├── middleware/
│   │   ├── handlers/
│   │   └── config/
│   ├── Dockerfile
│   ├── package.json
│   └── tsconfig.json
│
├── ml-services/              # Python ML Services
│   ├── nlp-service/
│   ├── url-service/
│   ├── visual-service/
│   ├── shared/               # Shared Python utilities
│   ├── Dockerfile
│   └── requirements.txt
│
├── core-services/            # Node.js Core Services
│   ├── detection-api/
│   ├── threat-intel/
│   ├── learning-pipeline/
│   ├── extension-api/
│   └── sandbox-service/
│
├── shared/                   # Shared code
│   ├── types/                # TypeScript type definitions
│   ├── database/             # Database schemas and migrations
│   ├── utils/                # Shared utilities
│   └── config/               # Configuration files
│
├── infrastructure/           # Infrastructure as Code
│   ├── terraform/
│   │   ├── modules/
│   │   ├── environments/
│   │   └── main.tf
│   └── cloudformation/       # Alternative CloudFormation templates
│
├── docker-compose.yml        # Local development
├── .env.example
└── README.md
```

### 2. AWS Infrastructure Setup (Terraform)

#### 2.1 VPC and Networking

**File**: `backend/infrastructure/terraform/modules/vpc/main.tf`

```hcl
# VPC with public and private subnets
# NAT Gateway for private subnet internet access
# Security groups for each service tier
# VPC endpoints for S3 and other AWS services
```

**Key Components**:
- VPC with CIDR 10.0.0.0/16
- 3 public subnets (multi-AZ)
- 3 private subnets (multi-AZ)
- Internet Gateway
- NAT Gateways (one per AZ)
- Security Groups:
  - API Gateway: Allow HTTPS (443) from internet
  - ECS Services: Allow internal communication only
  - RDS: Allow from ECS security group only
  - Redis: Allow from ECS security group only

#### 2.2 RDS PostgreSQL Setup

**File**: `backend/infrastructure/terraform/modules/rds/main.tf`

**Configuration**:
- Engine: PostgreSQL 15.x
- Instance: db.t3.medium (development), db.r5.large (production)
- Multi-AZ: Enabled for production
- Automated backups: 7 days retention
- Encryption: Enabled at rest
- Parameter group: Optimized for connection pooling
- Database name: `phishing_detection`

#### 2.3 ElastiCache Redis Setup

**File**: `backend/infrastructure/terraform/modules/redis/main.tf`

**Configuration**:
- Engine: Redis 7.x
- Node type: cache.t3.micro (dev), cache.r6g.large (prod)
- Cluster mode: Disabled (single node for now)
- Backup: Daily snapshots
- Encryption: In-transit and at-rest

#### 2.4 S3 Buckets

**File**: `backend/infrastructure/terraform/modules/s3/main.tf`

**Buckets**:
- `phishing-detection-models-{env}` - ML model storage
- `phishing-detection-training-data-{env}` - Training datasets
- `phishing-detection-logs-{env}` - Application logs
- `phishing-detection-artifacts-{env}` - CI/CD artifacts

**Policies**: 
- Versioning enabled
- Lifecycle policies for old versions
- Encryption: AES-256

#### 2.5 ECS/EKS Cluster Setup

**Option A: ECS (Simpler)**

**File**: `backend/infrastructure/terraform/modules/ecs/main.tf`

**Configuration**:
- Cluster: Fargate (serverless)
- Task definitions for each service
- Service discovery via Cloud Map
- Auto-scaling policies
- Load balancers (ALB) for public-facing services

**Option B: EKS (More flexible)**

**File**: `backend/infrastructure/terraform/modules/eks/main.tf`

**Configuration**:
- Kubernetes cluster (managed node groups)
- Ingress controller (ALB)
- Service mesh (optional: Istio/App Mesh)
- Horizontal Pod Autoscaler
- Cluster Autoscaler

**Recommendation**: Start with ECS Fargate, migrate to EKS if needed.

### 3. Docker Configuration

#### 3.1 API Gateway Dockerfile

**File**: `backend/api-gateway/Dockerfile`

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

#### 3.2 ML Services Dockerfile

**File**: `backend/ml-services/Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 3.3 Docker Compose for Local Development

**File**: `backend/docker-compose.yml`

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: phishing_detection
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  api-gateway:
    build: ./api-gateway
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://postgres:postgres@postgres:5432/phishing_detection
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
  mongo_data:
  redis_data:
```

### 4. CI/CD Pipeline Setup

#### 4.1 GitHub Actions Workflow

**File**: `.github/workflows/backend-ci.yml`

**Stages**:
1. **Lint & Test**: Run ESLint, TypeScript checks, unit tests
2. **Build**: Build Docker images
3. **Security Scan**: Scan Docker images for vulnerabilities
4. **Deploy Dev**: Deploy to development environment
5. **Integration Tests**: Run integration tests against dev
6. **Deploy Prod**: Manual approval required

**Key Features**:
- Matrix builds for multiple services
- Docker image tagging with commit SHA
- Push to AWS ECR
- Terraform plan/apply for infrastructure
- Rollback capability

### 5. Service Discovery & API Gateway

#### 5.1 API Gateway Configuration

**File**: `backend/api-gateway/src/config/gateway.ts`

**Features**:
- Route definitions for all services
- Request/response transformation
- Rate limiting per API key
- Request logging
- Error handling middleware
- CORS configuration

**Routes**:
```
/api/v1/detect/*          -> detection-api
/api/v1/intelligence/*    -> threat-intel
/api/v1/dashboard/*       -> detection-api
/api/v1/iocs/*            -> threat-intel
/ws/events                -> detection-api (WebSocket)
```

#### 5.2 Service Discovery

**Implementation**: AWS Cloud Map (ECS) or Kubernetes DNS (EKS)

**Service Names**:
- `detection-api.internal`
- `nlp-service.internal`
- `url-service.internal`
- `visual-service.internal`
- `threat-intel.internal`

### 6. Logging & Monitoring

#### 6.1 CloudWatch Integration

**Components**:
- CloudWatch Logs for application logs
- CloudWatch Metrics for custom metrics
- CloudWatch Alarms for alerting
- CloudWatch Dashboards for visualization

**Log Groups**:
- `/phishing-detection/api-gateway`
- `/phishing-detection/detection-api`
- `/phishing-detection/ml-services`
- `/phishing-detection/threat-intel`

#### 6.2 Prometheus & Grafana (Optional)

**File**: `backend/infrastructure/monitoring/prometheus.yml`

**Metrics**:
- Request latency (p50, p95, p99)
- Error rates
- ML model inference times
- Database query performance
- Cache hit rates
- Queue depths

### 7. Environment Configuration

#### 7.1 Environment Variables

**File**: `backend/.env.example`

```env
# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname
MONGODB_URL=mongodb://host:27017/dbname
REDIS_URL=redis://host:6379

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_BUCKET_MODELS=phishing-detection-models-dev
S3_BUCKET_TRAINING=phishing-detection-training-dev

# Services
NLP_SERVICE_URL=http://nlp-service:8000
URL_SERVICE_URL=http://url-service:8000
VISUAL_SERVICE_URL=http://visual-service:8000

# Security
JWT_SECRET=
API_KEY_ENCRYPTION_KEY=

# Monitoring
LOG_LEVEL=info
SENTRY_DSN=
```

#### 7.2 Configuration Management

**File**: `backend/shared/config/index.ts`

Use environment-specific config files:
- `config/development.ts`
- `config/staging.ts`
- `config/production.ts`

### 8. Authentication & Authorization Setup

#### 8.1 API Key Management

**Initial Implementation**:
- API keys stored in PostgreSQL
- Hashed using bcrypt
- Rate limiting per key
- Key rotation support

**Future**: Migrate to AWS API Gateway API Keys or Cognito

### 9. Testing Infrastructure

#### 9.1 Test Database Setup

**File**: `backend/shared/database/test-setup.ts`

- Separate test database
- Test fixtures and seeders
- Database cleanup between tests

#### 9.2 Integration Test Environment

- Docker Compose override for testing
- Mock external services
- Test data generators

## Deliverables Checklist

- [ ] Project directory structure created
- [ ] Terraform modules for AWS infrastructure
- [ ] Dockerfiles for all services
- [ ] Docker Compose for local development
- [ ] GitHub Actions CI/CD pipeline
- [ ] API Gateway routing configuration
- [ ] Service discovery setup
- [ ] CloudWatch logging configured
- [ ] Environment configuration files
- [ ] Basic authentication setup
- [ ] Documentation updated

## Testing

### Infrastructure Tests
- Terraform validate and plan
- Docker images build successfully
- Docker Compose starts all services
- Services can communicate via service discovery
- Database connections work
- Redis connections work

### CI/CD Tests
- Pipeline runs successfully
- Docker images pushed to ECR
- Terraform applies without errors
- Services deploy to ECS/EKS

## Next Steps

After completing Phase 1:
1. Verify all infrastructure is running
2. Test service-to-service communication
3. Set up monitoring dashboards
4. Document deployment procedures
5. Proceed to Phase 2: Database Schema & Data Models

## Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
