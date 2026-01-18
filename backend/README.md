# Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System - Backend

This directory contains the backend infrastructure and services for the Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System.

## Project Structure

```
backend/
├── api-gateway/          # Node.js/TypeScript API Gateway
├── ml-services/          # Python ML Services (NLP, URL, Visual)
├── core-services/        # Node.js Core Services
├── shared/               # Shared code and utilities
├── infrastructure/       # Infrastructure as Code (Terraform)
└── docker-compose.yml    # Local development setup
```

## Prerequisites

- Node.js 20+
- Python 3.11+
- Docker Desktop
- Terraform CLI
- AWS Account with appropriate permissions

## Quick Start

### Local Development

1. **Copy environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Start services with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Install API Gateway dependencies:**
   ```bash
   cd api-gateway
   npm install
   ```

4. **Run API Gateway in development mode:**
   ```bash
   npm run dev
   ```

The API Gateway will be available at `http://localhost:3000`

### Health Check

```bash
curl http://localhost:3000/health
```

## Infrastructure Deployment

### Prerequisites

1. Configure AWS credentials:
   ```bash
   aws configure
   ```

2. Set up Terraform backend (S3 bucket for state):
   - Create an S3 bucket for Terraform state
   - Update `backend/infrastructure/terraform/main.tf` with your backend configuration

### Deploy Infrastructure

1. **Navigate to Terraform directory:**
   ```bash
   cd infrastructure/terraform
   ```

2. **Copy environment variables:**
   ```bash
   cp environments/dev.tfvars.example environments/dev.tfvars
   # Edit dev.tfvars with your values
   ```

3. **Initialize Terraform:**
   ```bash
   terraform init
   ```

4. **Plan infrastructure:**
   ```bash
   terraform plan -var-file=environments/dev.tfvars
   ```

5. **Apply infrastructure:**
   ```bash
   terraform apply -var-file=environments/dev.tfvars
   ```

## Services

### API Gateway

- **Port:** 3000
- **Routes:**
  - `/health` - Health check endpoint
  - `/api/v1/detect/*` - Detection API routes
  - `/api/v1/intelligence/*` - Threat intelligence routes
  - `/api/v1/dashboard/*` - Dashboard API routes
  - `/api/v1/iocs/*` - IOC management routes

### ML Services

- **NLP Service:** Port 8000
- **URL Service:** Port 8000
- **Visual Service:** Port 8000

### Core Services

- **Detection API:** Port 3001
- **Threat Intel:** Port 3002
- **Learning Pipeline:** Background service
- **Extension API:** Port 3003
- **Sandbox Service:** Port 3004

## Environment Variables

See `.env.example` for all available environment variables.

Key variables:
- `DATABASE_URL` - PostgreSQL connection string
- `MONGODB_URL` - MongoDB connection string
- `REDIS_URL` - Redis connection string
- `AWS_REGION` - AWS region
- `JWT_SECRET` - JWT signing secret
- `LOG_LEVEL` - Logging level (info, debug, warn, error)

## CI/CD

The project uses GitHub Actions for CI/CD. See `.github/workflows/backend-ci.yml` for the pipeline configuration.

Pipeline stages:
1. Lint & Test
2. Build Docker Images
3. Security Scan
4. Terraform Plan (on PRs)
5. Deploy to Development (on develop branch)
6. Integration Tests
7. Deploy to Production (on main branch, manual approval)

## Testing

### Unit Tests

```bash
cd api-gateway
npm test
```

### Integration Tests

```bash
# Run integration tests against local services
npm run test:integration
```

## Monitoring

- **CloudWatch Logs:** Application logs are sent to CloudWatch
- **CloudWatch Metrics:** Custom metrics for monitoring
- **Health Checks:** `/health` endpoint for service health

## Security

- API keys are required for all API endpoints (except `/health`)
- Rate limiting is enforced per API key
- All traffic is encrypted in transit
- Database and cache encryption at rest

## Documentation

- [Phase 1: Infrastructure](../docs/phases/phase-1-infrastructure.md)
- [Phase 2: Database](../docs/phases/phase-2-database.md)
- [Phase 3: NLP Service](../docs/phases/phase-3-nlp-service.md)
- [Phase 4: URL Service](../docs/phases/phase-4-url-service.md)

## Troubleshooting

### Services won't start

1. Check Docker is running: `docker ps`
2. Check logs: `docker-compose logs`
3. Verify environment variables are set correctly

### Database connection issues

1. Verify PostgreSQL is running: `docker-compose ps postgres`
2. Check connection string in `.env`
3. Test connection: `psql $DATABASE_URL`

### Terraform errors

1. Verify AWS credentials: `aws sts get-caller-identity`
2. Check Terraform version: `terraform version`
3. Validate configuration: `terraform validate`

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linters
4. Submit a pull request

## License

[Your License Here]
