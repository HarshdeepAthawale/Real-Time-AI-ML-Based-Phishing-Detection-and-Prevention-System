# Real-Time AI/ML-Based Phishing Detection and Prevention System

A comprehensive, enterprise-grade phishing detection and prevention platform leveraging advanced machine learning models, real-time analysis, and threat intelligence integration to protect organizations from sophisticated phishing attacks.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Components](#system-components)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Monitoring & Observability](#monitoring--observability)
- [Security](#security)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The Real-Time AI/ML-Based Phishing Detection and Prevention System is a microservices-based platform designed to detect and prevent phishing attacks through multi-layered analysis. The system combines natural language processing, graph neural networks, computer vision, and threat intelligence to provide comprehensive protection against evolving phishing threats.

### Key Capabilities

- **Multi-Modal Detection**: Analyzes text content, URLs, domains, and visual elements simultaneously
- **Real-Time Processing**: Sub-50ms latency for threat detection and classification
- **Advanced ML Models**: Fine-tuned BERT/RoBERTa for NLP, Graph Neural Networks for URL analysis, and CNNs for visual pattern recognition
- **Threat Intelligence Integration**: Real-time IOC matching with MISP and AlienVault OTX feeds
- **Continuous Learning**: Automated model retraining pipeline with drift detection and A/B testing
- **Browser Extension**: Real-time protection for users browsing the web
- **Sandbox Analysis**: Dynamic behavioral analysis of suspicious links and attachments
- **Scalable Architecture**: Cloud-native design with auto-scaling and high availability

### Target Use Cases

- Enterprise email security and filtering
- Browser-based threat protection
- API-based threat analysis for security tools
- Real-time threat monitoring and alerting
- Security research and threat intelligence

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js 16)                        │
│  Threat Dashboard | Real-time Monitor | Threat Intelligence     │
└────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    API Gateway (Node.js/TypeScript)              │
│  Routing | Authentication | Rate Limiting | Request Logging     │
└───────┬───────────────┬───────────────┬──────────────────────────┘
        │               │               │
┌───────▼──────┐  ┌─────▼──────┐  ┌───▼──────────────┐
│ Detection API│  │Threat Intel│  │  Extension API   │
│  (Orchestrator)│  │  Service   │  │   Service        │
└───────┬──────┘  └────────────┘  └──────────────────┘
        │
    ┌───┴───┬──────────┬──────────┬──────────┐
    │       │          │          │          │
┌───▼───┐ ┌─▼───┐ ┌───▼───┐ ┌────▼────┐ ┌────▼────┐
│ NLP   │ │URL  │ │Visual │ │Sandbox  │ │Learning │
│Service│ │Svc  │ │Svc    │ │Service  │ │Pipeline │
└───┬───┘ └─┬───┘ └───┬───┘ └────┬────┘ └────┬────┘
    │       │          │          │          │
    └───────┴──────────┴──────────┴──────────┘
            │          │          │
    ┌───────▼──────────▼──────────▼──────────┐
    │     PostgreSQL | MongoDB | Redis      │
    └────────────────────────────────────────┘
```

### Microservices Overview

The system consists of 8 core services:

**ML Services (Python/FastAPI):**
- **NLP Service**: Text analysis using transformer models
- **URL Service**: Domain and URL analysis using Graph Neural Networks
- **Visual Service**: Visual pattern recognition using CNNs

**Core Services (Node.js/TypeScript):**
- **API Gateway**: Entry point with authentication and routing
- **Detection API**: Orchestrates ML services for threat detection
- **Threat Intelligence Service**: IOC management and feed synchronization
- **Extension API**: Browser extension backend integration
- **Sandbox Service**: Dynamic analysis of suspicious content
- **Learning Pipeline**: Automated model training and deployment

### Data Flow

1. **Request Flow**: Client → API Gateway → Detection API → ML Services
2. **Analysis Flow**: Parallel processing across NLP, URL, and Visual services
3. **Decision Flow**: Ensemble decision engine aggregates results
4. **Storage Flow**: Results stored in PostgreSQL (metadata) and MongoDB (analysis data)
5. **Cache Flow**: Frequently accessed data cached in Redis
6. **Intelligence Flow**: IOC matching against threat intelligence feeds
7. **Feedback Flow**: User feedback collected for continuous learning

## Features

### Core Detection Features

**Text Analysis (NLP Service)**
- Phishing detection using fine-tuned BERT/RoBERTa models
- AI-generated content detection
- Urgency and sentiment analysis
- Social engineering indicator identification
- Email parsing and header analysis
- Multi-language support

**URL and Domain Analysis (URL Service)**
- Graph Neural Network-based domain relationship analysis
- Redirect chain tracking and analysis
- Homoglyph and typosquatting detection
- WHOIS and DNS record analysis
- SSL certificate validation
- Domain reputation scoring
- Subdomain enumeration

**Visual Analysis (Visual Service)**
- CNN-based brand impersonation detection
- DOM structure analysis
- Visual similarity matching
- Logo and form field detection
- CSS pattern analysis
- Screenshot-based comparison

### Real-Time Features

- **Sub-50ms Detection Latency**: Optimized for real-time threat detection
- **WebSocket Support**: Real-time event streaming to clients
- **Parallel Processing**: Simultaneous analysis across multiple ML services
- **Caching Strategy**: Redis-based caching for frequently accessed data
- **Async Processing**: Background jobs for heavy analysis tasks

### Threat Intelligence

- **IOC Management**: Comprehensive Indicators of Compromise database
- **Feed Integration**: MISP and AlienVault OTX synchronization
- **Bloom Filter Lookups**: Fast IOC matching with probabilistic data structures
- **Custom Feeds**: Support for custom threat intelligence sources
- **IOC Enrichment**: Automatic enrichment with metadata and context

### Continuous Learning

- **Automated Training Pipeline**: Scheduled model retraining from new data
- **Drift Detection**: Automatic detection of model performance degradation
- **A/B Testing**: Model version comparison and gradual rollout
- **Feedback Loop**: User feedback integration for model improvement
- **Feature Store**: Centralized feature management for training

### Browser Extension

- **Real-Time URL Checking**: Instant threat detection while browsing
- **Email Scanning**: Integration with email clients for message analysis
- **Privacy-Preserving**: Local caching and minimal data transmission
- **Blocking Mechanisms**: Automatic blocking of confirmed threats
- **User Reporting**: Easy reporting of suspicious content

### Sandbox Analysis

- **Dynamic Analysis**: Behavioral analysis of links and attachments
- **Multi-Engine Support**: Integration with Cuckoo, Any.run, and custom sandboxes
- **Result Correlation**: Integration with detection signals
- **Automated Submission**: Queue-based processing for sandbox jobs
- **Comprehensive Reporting**: Detailed behavioral analysis reports

## Technology Stack

### Frontend

- **Framework**: Next.js 16.0.10 (App Router)
- **UI Library**: React 19.2.0
- **Language**: TypeScript 5.x
- **Styling**: Tailwind CSS 4.1.9
- **Components**: Radix UI (comprehensive component library)
- **Charts**: Recharts 2.15.4
- **State Management**: React Hooks
- **Theming**: next-themes (dark/light mode support)

### Backend Services

**API Gateway & Core Services:**
- **Runtime**: Node.js 20+
- **Framework**: Express.js 4.18.2
- **Language**: TypeScript 5.3.3
- **Authentication**: API key-based with bcrypt
- **Rate Limiting**: Custom middleware with Redis
- **WebSocket**: Socket.io 4.5.4

**ML Services:**
- **Runtime**: Python 3.11+
- **Framework**: FastAPI 0.104.1
- **Server**: Uvicorn 0.24.0
- **Validation**: Pydantic 2.5.0

### Machine Learning & AI

- **NLP**: Transformers 4.35.0, Sentence Transformers 2.2.2
- **Deep Learning**: PyTorch 2.1.0
- **Graph Networks**: PyTorch Geometric 2.4.0
- **Computer Vision**: Torchvision 0.16.0, OpenCV 4.8.1
- **ML Utilities**: scikit-learn 1.3.2, NumPy 1.24.3
- **NLP Tools**: NLTK 3.8.1, spaCy 3.7.2

### Databases

- **PostgreSQL 15**: Primary relational database (20 tables)
- **MongoDB 7**: Document store for ML analysis results
- **Redis 7**: Caching, queues (BullMQ), rate limiting

### Infrastructure

- **Cloud Provider**: AWS (ap-south-1)
- **Infrastructure as Code**: Terraform 1.5.0
- **Containerization**: Docker, Docker Compose
- **Orchestration**: AWS ECS Fargate
- **Load Balancing**: Application Load Balancer
- **Service Discovery**: AWS Cloud Map
- **Storage**: AWS S3 (models, training data, logs)
- **CI/CD**: GitHub Actions

### DevOps & Monitoring

- **CI/CD**: GitHub Actions workflows
- **Logging**: CloudWatch Logs, Winston
- **Monitoring**: CloudWatch Metrics
- **Security Scanning**: Trivy
- **Version Control**: Git

## System Components

### Phase 1: Infrastructure & Architecture

**Status**: Complete

The foundation layer provides the infrastructure and architectural patterns for the entire system.

**Components:**
- VPC with public and private subnets (multi-AZ)
- RDS PostgreSQL 15.x with multi-AZ support
- ElastiCache Redis 7.x for caching and queues
- S3 buckets for models, training data, logs, and artifacts
- ECS Fargate cluster with Application Load Balancer
- Security groups and network ACLs
- Service discovery via AWS Cloud Map
- CloudWatch logging for all services
- Docker containerization for all services
- Docker Compose for local development
- Terraform modules for infrastructure provisioning
- GitHub Actions CI/CD pipeline

**Key Features:**
- Multi-AZ deployment for high availability
- Auto-scaling policies for services
- Encrypted storage and communication
- Health check endpoints for all services
- Environment-specific configurations

### Phase 2: Database Schema & Data Models

**Status**: Complete

Comprehensive database schemas supporting all system functionality.

**PostgreSQL Schema (20 tables):**

**Users & Organizations:**
- `organizations`: Organization management with plan tiers
- `users`: User accounts with role-based access
- `api_keys`: API key management with rate limiting

**Threats & Detections:**
- `threats`: Threat records with severity and type classification
- `detections`: Detection results with confidence scores
- `threat_indicators`: Individual threat indicators
- `detection_feedback`: User feedback for model improvement

**Domains & URLs:**
- `domains`: Domain records with reputation scores
- `urls`: URL analysis results
- `domain_relationships`: Graph relationships for GNN analysis

**ML Models & Training:**
- `ml_models`: Model metadata and configuration
- `model_versions`: Version tracking for models
- `training_jobs`: Training job history and status
- `model_performance`: Performance metrics and evaluation

**Threat Intelligence:**
- `threat_intelligence_feeds`: Feed configuration
- `iocs`: Indicators of Compromise
- `ioc_matches`: IOC match records

**Email Messages:**
- `email_messages`: Email metadata
- `email_headers`: Email header analysis

**Sandbox Analyses:**
- `sandbox_analyses`: Sandbox analysis results

**MongoDB Collections:**
- `email_content`: Email body and NLP analysis results
- `url_analysis`: URL graph and GNN analysis data
- `visual_analysis`: Visual and CNN analysis results

**Redis Structures:**
- Cache keys for URL/domain reputation, IOC lookups, model inference
- Queue keys for detection jobs, sandbox jobs, training jobs, threat intel sync
- Rate limiting keys per API key

**Key Features:**
- Comprehensive indexing for query optimization
- Foreign key constraints with appropriate cascade rules
- JSONB columns for flexible metadata
- UUID primary keys for distributed systems
- Soft delete support with `deleted_at` columns
- Audit trails with timestamps

### Phase 3: NLP Text Analysis Service

**Status**: Complete

Transformer-based NLP service for semantic analysis of email and SMS content.

**Models:**
- Fine-tuned BERT/RoBERTa for phishing detection
- AI-generated content detector
- Urgency analyzer
- Sentiment analyzer
- Social engineering indicator detector

**Features:**
- Text normalization and preprocessing
- Email parsing and header extraction
- Feature extraction (linguistic, semantic, structural)
- Multi-language support
- Confidence scoring
- Processing time tracking

**Endpoints:**
- `POST /api/v1/analyze-text`: Analyze plain text content
- `POST /api/v1/analyze-email`: Analyze raw email content
- `POST /api/v1/detect-ai-content`: Detect AI-generated content
- `GET /health`: Health check endpoint

**Performance:**
- Average processing time: <30ms
- Supports batch processing
- Redis caching for frequently analyzed content

### Phase 4: URL/Domain Analysis Service

**Status**: Complete

Graph Neural Network-based service for URL and domain analysis.

**Models:**
- GNN classifier for domain relationship analysis
- Reputation scoring model
- Homoglyph detection model

**Features:**
- URL parsing and normalization
- Domain extraction and analysis
- WHOIS data retrieval and analysis
- DNS record analysis (A, AAAA, MX, TXT, NS)
- SSL certificate validation
- Redirect chain tracking
- Link extraction from web pages
- Graph construction for domain relationships
- Homoglyph and typosquatting detection
- Subdomain enumeration

**Endpoints:**
- `POST /api/v1/analyze-url`: Analyze URL and domain
- `POST /api/v1/analyze-domain`: Analyze domain only
- `POST /api/v1/track-redirects`: Track redirect chain
- `GET /health`: Health check endpoint

**Performance:**
- Average processing time: <40ms
- Graph analysis with NetworkX
- Caching for domain reputation scores

### Phase 5: Visual/Structural Analysis Service

**Status**: Complete

CNN-based service for visual pattern recognition and brand impersonation detection.

**Models:**
- CNN classifier for brand impersonation
- Visual similarity matching model
- Logo detection model

**Features:**
- Headless browser rendering (Playwright)
- Screenshot capture and processing
- DOM structure analysis
- Form field detection and analysis
- CSS pattern analysis
- Logo detection and matching
- Visual similarity scoring
- Image processing with OpenCV and PIL

**Endpoints:**
- `POST /api/v1/analyze-page`: Analyze webpage visually
- `POST /api/v1/analyze-image`: Analyze uploaded image
- `POST /api/v1/compare-visual`: Compare two pages/images
- `GET /health`: Health check endpoint

**Performance:**
- Average processing time: <100ms (includes rendering)
- Screenshot caching in S3
- Async processing for heavy analysis

### Phase 6: Real-Time Detection API

**Status**: Complete

High-performance orchestration service for real-time threat detection.

**Features:**
- Parallel ML service orchestration
- Ensemble decision engine with weighted confidence
- WebSocket support for real-time event streaming
- Redis caching for detection results
- Event streaming to connected clients
- Sub-50ms latency target
- Request queuing with BullMQ
- Error handling and retry logic

**Endpoints:**
- `POST /api/v1/detect`: Multi-modal threat detection
- `POST /api/v1/detect/email`: Email-specific detection
- `POST /api/v1/detect/url`: URL-specific detection
- `GET /api/v1/detections/:id`: Get detection result
- `WS /ws/detections`: WebSocket connection for real-time updates
- `GET /health`: Health check endpoint

**Decision Engine:**
- Weighted confidence aggregation
- Source-specific confidence thresholds
- Threat severity classification
- False positive reduction strategies

### Phase 7: Threat Intelligence Integration Service

**Status**: Complete

IOC management and threat intelligence feed integration.

**Features:**
- MISP API integration
- AlienVault OTX integration
- Custom feed support
- IOC CRUD operations
- Fast IOC matching with Bloom filters
- Feed synchronization scheduler
- IOC enrichment with metadata
- Normalization of IOC formats

**Endpoints:**
- `GET /api/v1/iocs`: List IOCs with filtering
- `POST /api/v1/iocs`: Create new IOC
- `GET /api/v1/iocs/:id`: Get IOC details
- `PUT /api/v1/iocs/:id`: Update IOC
- `DELETE /api/v1/iocs/:id`: Delete IOC
- `POST /api/v1/iocs/match`: Match IOCs against input
- `POST /api/v1/feeds/sync`: Manual feed synchronization
- `GET /health`: Health check endpoint

**Synchronization:**
- Scheduled sync jobs (configurable intervals)
- Incremental updates
- Conflict resolution
- Feed health monitoring

### Phase 8: Continuous Learning & Model Training Pipeline

**Status**: Complete

Automated ML pipeline for incremental learning and model improvement.

**Features:**
- Data collection from user feedback
- Feature store management
- Training job orchestration
- Model validation and evaluation
- Drift detection
- A/B testing framework
- Model deployment automation
- Performance monitoring

**Components:**
- Data collector service
- Feature store service
- Training orchestrator
- Model validator
- Drift detector
- Deployment service

**Training Pipeline:**
- Scheduled training jobs
- Incremental learning support
- Hyperparameter optimization
- Cross-validation
- Model versioning
- Automated deployment to staging/production

**Monitoring:**
- Model performance tracking
- Drift detection alerts
- Training job status
- Resource utilization

### Phase 9: Browser Extension Backend & Edge Integration

**Status**: Complete

Lightweight backend services for browser extension integration.

**Features:**
- Real-time URL checking
- Email scanning integration
- Privacy-preserving analysis
- Local caching support
- Extension authentication
- Reporting endpoints
- Configuration management

**Endpoints:**
- `POST /api/v1/extension/check-url`: Check URL in real-time
- `POST /api/v1/extension/scan-email`: Scan email content
- `POST /api/v1/extension/report`: Report suspicious content
- `GET /api/v1/extension/config`: Get extension configuration
- `GET /health`: Health check endpoint

**Privacy Features:**
- Minimal data transmission
- Local caching of safe URLs
- Configurable privacy levels
- No PII collection

### Phase 10: Sandbox Integration & Advanced Threat Analysis

**Status**: Complete

Dynamic sandbox environment for behavioral analysis.

**Features:**
- Multi-sandbox support (Cuckoo, Any.run, custom)
- File analysis and submission
- Behavioral analysis
- Result processing and correlation
- Job queue management
- Automated submission workflows

**Endpoints:**
- `POST /api/v1/sandbox/submit`: Submit file/URL for analysis
- `GET /api/v1/sandbox/analysis/:id`: Get analysis results
- `GET /api/v1/sandbox/status/:id`: Get analysis status
- `POST /api/v1/sandbox/correlate`: Correlate with detections
- `GET /health`: Health check endpoint

**Analysis Capabilities:**
- Network activity analysis
- File system monitoring
- Process behavior tracking
- Registry changes
- Screenshot capture
- Signature detection
- Malware scoring

## Installation & Setup

### Prerequisites

- **Node.js**: 20.0.0 or higher
- **Python**: 3.11.0 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **Terraform**: 1.5.0 or higher (for infrastructure deployment)
- **AWS CLI**: Configured with appropriate credentials
- **Git**: For version control

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System.git
   cd Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System
   ```

2. **Set up environment variables:**
   ```bash
   cd backend
   cp env.template .env
   # Edit .env with your configuration
   ```

3. **Start services with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

   This will start:
   - PostgreSQL (port 5432)
   - MongoDB (port 27017)
   - Redis (port 6379)
   - All backend services

4. **Initialize databases:**
   ```bash
   cd backend/shared/database
   npm run migrate
   npm run seed  # Optional: seed with sample data
   ```

5. **Install frontend dependencies:**
   ```bash
   cd ../..
   npm install
   ```

6. **Start frontend development server:**
   ```bash
   npm run dev
   ```

7. **Install backend service dependencies:**
   ```bash
   cd backend/api-gateway
   npm install
   npm run dev
   ```

### Environment Configuration

Key environment variables:

**Backend Services:**
```bash
DATABASE_URL=postgresql://postgres:password@localhost:5432/phishing_detection
MONGODB_URL=mongodb://localhost:27017/phishing_detection
REDIS_URL=redis://localhost:6379
NODE_ENV=development
LOG_LEVEL=info
JWT_SECRET=your-secret-key
```

**ML Services:**
```bash
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://postgres:password@localhost:5432/phishing_detection
MONGODB_URL=mongodb://localhost:27017/phishing_detection
MODEL_PATH=/app/models
LOG_LEVEL=info
```

**AWS Configuration:**
```bash
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_MODELS=your-models-bucket
S3_BUCKET_TRAINING=your-training-bucket
```

### Database Initialization

1. **Run migrations:**
   ```bash
   cd backend/shared/database
   npm run migrate
   ```

2. **Verify database setup:**
   ```bash
   npm run verify
   ```

3. **Seed sample data (optional):**
   ```bash
   npm run seed
   ```

## Usage

### API Authentication

All API endpoints (except `/health`) require authentication via API key.

**Header Format:**
```
X-API-Key: your-api-key-here
```

**Generate API Key:**
```bash
curl -X POST http://localhost:3000/api/v1/auth/api-keys \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My API Key",
    "permissions": ["detect", "read"]
  }'
```

### Example API Requests

**Detect Phishing in Text:**
```bash
curl -X POST http://localhost:3000/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "Urgent: Your account has been suspended. Click here to verify.",
    "include_features": true
  }'
```

**Analyze URL:**
```bash
curl -X POST http://localhost:3000/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "url": "https://example.com/verify-account",
    "legitimate_domain": "example.com"
  }'
```

**Analyze Email:**
```bash
curl -X POST http://localhost:3000/api/v1/detect/email \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "email_content": "From: bank@example.com\nSubject: Urgent Security Alert\n\nYour account needs verification...",
    "include_features": true
  }'
```

### WebSocket Connection

**Connect to real-time detection stream:**
```javascript
const socket = io('http://localhost:3001', {
  auth: {
    apiKey: 'your-api-key'
  }
});

socket.on('detection', (data) => {
  console.log('New detection:', data);
});
```

### Browser Extension

1. **Install Extension:**
   - Chrome: Load unpacked from `extensions/chrome/`
   - Firefox: Load temporary from `extensions/firefox/`
   - Edge: Load unpacked from `extensions/edge/`

2. **Configure Extension:**
   - Set API endpoint URL
   - Configure privacy settings
   - Enable/disable features

3. **Usage:**
   - Extension automatically checks URLs while browsing
   - Click extension icon to view threat status
   - Report suspicious content via extension

### Dashboard Access

1. **Access Dashboard:**
   ```
   http://localhost:3000
   ```

2. **Login:**
   - Use your organization credentials
   - Or API key for programmatic access

3. **Features:**
   - Real-time threat monitoring
   - Threat intelligence feed
   - Detection history
   - Model performance metrics
   - Configuration management

## Deployment

### AWS Infrastructure Setup

1. **Configure Terraform:**
   ```bash
   cd backend/infrastructure/terraform
   cp environments/dev.tfvars.example environments/dev.tfvars
   # Edit dev.tfvars with your values
   ```

2. **Initialize Terraform:**
   ```bash
   terraform init
   ```

3. **Plan Infrastructure:**
   ```bash
   terraform plan -var-file=environments/dev.tfvars
   ```

4. **Apply Infrastructure:**
   ```bash
   terraform apply -var-file=environments/dev.tfvars
   ```

5. **Get Outputs:**
   ```bash
   terraform output
   ```

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment.

**Pipeline Stages:**
1. **Lint & Test**: Code quality checks and unit tests
2. **Build Docker Images**: Build images for all services
3. **Security Scan**: Trivy vulnerability scanning
4. **Terraform Plan**: Infrastructure changes preview
5. **Deploy Dev**: Automatic deployment to development environment
6. **Integration Tests**: End-to-end testing
7. **Deploy Prod**: Manual approval for production deployment

**Workflow Files:**
- `.github/workflows/backend-ci.yml`: Backend services CI/CD
- `.github/workflows/frontend-ci.yml`: Frontend CI/CD

### Production Considerations

**Scaling:**
- ECS auto-scaling based on CPU/memory metrics
- Application Load Balancer for traffic distribution
- Redis cluster mode for high availability
- RDS read replicas for read scaling

**Monitoring:**
- CloudWatch Logs for all services
- CloudWatch Metrics for performance tracking
- CloudWatch Alarms for alerting
- Health check endpoints for load balancer

**Security:**
- VPC with private subnets for services
- Security groups with least privilege
- Encrypted storage (at rest and in transit)
- API key rotation policies
- Regular security audits

**Backup:**
- RDS automated backups (7-day retention)
- S3 versioning for model files
- MongoDB backup scripts
- Disaster recovery procedures

## API Documentation

### Main Endpoints

**API Gateway Base URL:** `http://localhost:3000`

**Detection Endpoints:**
- `POST /api/v1/detect` - Multi-modal threat detection
- `POST /api/v1/detect/email` - Email-specific detection
- `POST /api/v1/detect/url` - URL-specific detection
- `GET /api/v1/detections/:id` - Get detection result

**Threat Intelligence Endpoints:**
- `GET /api/v1/iocs` - List IOCs
- `POST /api/v1/iocs` - Create IOC
- `GET /api/v1/iocs/:id` - Get IOC details
- `POST /api/v1/iocs/match` - Match IOCs

**Extension Endpoints:**
- `POST /api/v1/extension/check-url` - Check URL
- `POST /api/v1/extension/scan-email` - Scan email
- `POST /api/v1/extension/report` - Report threat

**Sandbox Endpoints:**
- `POST /api/v1/sandbox/submit` - Submit for analysis
- `GET /api/v1/sandbox/analysis/:id` - Get results

### Request/Response Formats

**Detection Request:**
```json
{
  "text": "string (optional)",
  "url": "string (optional)",
  "email_content": "string (optional)",
  "image": "base64 string (optional)",
  "include_features": boolean,
  "legitimate_domain": "string (optional)",
  "legitimate_url": "string (optional)"
}
```

**Detection Response:**
```json
{
  "is_phishing": boolean,
  "confidence": float,
  "sources": {
    "nlp": { ... },
    "url": { ... },
    "visual": { ... }
  },
  "overall_confidence": float,
  "timestamp": "ISO 8601 string"
}
```

### Rate Limiting

- Default: 100 requests per minute per API key
- Configurable per organization
- Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

### Error Responses

**Standard Error Format:**
```json
{
  "error": "Error code",
  "message": "Human-readable message",
  "details": { ... }
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `429`: Too Many Requests
- `500`: Internal Server Error
- `503`: Service Unavailable

## Development

### Project Structure

```
.
├── app/                    # Next.js frontend app directory
├── components/             # React components
├── backend/
│   ├── api-gateway/       # API Gateway service
│   ├── ml-services/       # ML services (Python)
│   │   ├── nlp-service/
│   │   ├── url-service/
│   │   └── visual-service/
│   ├── core-services/     # Core services (Node.js)
│   │   ├── detection-api/
│   │   ├── threat-intel/
│   │   ├── extension-api/
│   │   ├── sandbox-service/
│   │   └── learning-pipeline/
│   ├── shared/            # Shared code
│   │   ├── database/      # Database models and migrations
│   │   ├── types/         # TypeScript types
│   │   └── utils/         # Utilities
│   └── infrastructure/    # Infrastructure as Code
│       └── terraform/
├── docs/                  # Documentation
│   └── phases/           # Phase documentation
└── extensions/           # Browser extensions
```

### Contributing

1. **Create Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes:**
   - Follow code style guidelines
   - Write tests for new features
   - Update documentation

3. **Run Tests:**
   ```bash
   npm test
   ```

4. **Run Linters:**
   ```bash
   npm run lint
   ```

5. **Commit Changes:**
   ```bash
   git commit -m "feat: add new feature"
   ```

6. **Push and Create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Code Style

**TypeScript/JavaScript:**
- ESLint configuration
- Prettier for formatting
- TypeScript strict mode

**Python:**
- Black for formatting
- Flake8 for linting
- Type hints required

### Testing

**Unit Tests:**
```bash
# Backend services
cd backend/api-gateway
npm test

# ML services
cd backend/ml-services/nlp-service
pytest
```

**Integration Tests:**
```bash
npm run test:integration
```

**E2E Tests:**
```bash
npm run test:e2e
```

## Monitoring & Observability

### Logging

**Structured Logging:**
- Winston for Node.js services
- Python logging for ML services
- JSON format for log aggregation
- Log levels: DEBUG, INFO, WARN, ERROR

**Log Aggregation:**
- CloudWatch Logs for AWS deployment
- Log groups per service
- Retention: 30 days (configurable)

### Metrics

**Application Metrics:**
- Request latency
- Request count
- Error rates
- Detection accuracy
- Model inference time

**Infrastructure Metrics:**
- CPU utilization
- Memory usage
- Network I/O
- Database connections
- Cache hit rates

**Monitoring Tools:**
- CloudWatch Metrics
- Custom dashboards
- Alerting rules

### Health Checks

**Health Check Endpoints:**
- All services: `GET /health`
- Database connectivity
- External service availability
- Model loading status

**Health Check Response:**
```json
{
  "status": "healthy",
  "timestamp": "ISO 8601 string",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "models": "loaded"
  }
}
```

### Alerting

**Alert Conditions:**
- Service downtime
- High error rates
- Performance degradation
- Model drift detection
- Resource exhaustion

**Alert Channels:**
- CloudWatch Alarms
- Email notifications
- Slack integration (configurable)

## Security

### Authentication & Authorization

**API Key Authentication:**
- Bcrypt hashing for API keys
- Key prefix for identification
- Rate limiting per key
- Key expiration support
- Key revocation

**Role-Based Access Control:**
- Organization-level permissions
- User roles: admin, user, viewer
- Resource-level access control

### Data Encryption

**At Rest:**
- RDS encryption enabled
- S3 server-side encryption (AES-256)
- MongoDB encryption
- Redis encryption

**In Transit:**
- TLS 1.2+ for all connections
- HTTPS for API endpoints
- Encrypted database connections
- Redis TLS support

### API Security

**Security Headers:**
- Helmet.js for security headers
- CORS configuration
- Content Security Policy
- XSS protection

**Input Validation:**
- Pydantic models for Python
- Zod schemas for TypeScript
- SQL injection prevention
- XSS prevention

**Rate Limiting:**
- Per API key limits
- IP-based rate limiting
- DDoS protection

### Best Practices

- Regular security audits
- Dependency vulnerability scanning
- Secrets management (AWS Secrets Manager)
- Least privilege access
- Regular key rotation
- Security incident response plan

## Performance

### Latency Targets

- **Detection API**: <50ms (p95)
- **NLP Service**: <30ms (p95)
- **URL Service**: <40ms (p95)
- **Visual Service**: <100ms (p95)
- **IOC Matching**: <10ms (p95)

### Scalability

**Horizontal Scaling:**
- Stateless services for easy scaling
- Load balancer distribution
- Auto-scaling based on metrics

**Vertical Scaling:**
- Database instance scaling
- Cache cluster scaling
- Model serving optimization

### Caching Strategies

**Redis Caching:**
- URL reputation: 1 hour TTL
- Domain reputation: 6 hours TTL
- Model inference: 24 hours TTL
- IOC lookups: 1 hour TTL

**Cache Invalidation:**
- Time-based expiration
- Event-based invalidation
- Manual cache clearing

### Optimization

**Database:**
- Indexed queries
- Connection pooling
- Query optimization
- Read replicas for scaling

**ML Services:**
- Model quantization
- Batch processing
- GPU acceleration (optional)
- Model caching

**API:**
- Response compression
- Parallel processing
- Async operations
- Request batching

## Troubleshooting

### Common Issues

**Services Won't Start:**
1. Check Docker is running: `docker ps`
2. Check logs: `docker-compose logs [service-name]`
3. Verify environment variables
4. Check port availability

**Database Connection Issues:**
1. Verify PostgreSQL is running: `docker-compose ps postgres`
2. Check connection string in `.env`
3. Test connection: `psql $DATABASE_URL`
4. Check network connectivity

**ML Model Loading Errors:**
1. Verify model files exist in S3/local storage
2. Check model path configuration
3. Verify model file permissions
4. Check disk space

**High Latency:**
1. Check Redis cache hit rates
2. Monitor database query performance
3. Review service logs for errors
4. Check network latency
5. Verify auto-scaling is working

**Terraform Errors:**
1. Verify AWS credentials: `aws sts get-caller-identity`
2. Check Terraform version: `terraform version`
3. Validate configuration: `terraform validate`
4. Check state file conflicts

### Debugging Tips

**Enable Debug Logging:**
```bash
LOG_LEVEL=debug docker-compose up
```

**Check Service Health:**
```bash
curl http://localhost:3000/health
curl http://localhost:8001/health  # NLP Service
curl http://localhost:8002/health  # URL Service
```

**View Service Logs:**
```bash
docker-compose logs -f [service-name]
```

**Database Queries:**
```bash
psql $DATABASE_URL -c "SELECT * FROM detections LIMIT 10;"
```

**Redis Inspection:**
```bash
redis-cli
> KEYS *
> GET url:reputation:example.com
```

### Support Resources

- **Documentation**: See `docs/` directory
- **Phase Documentation**: `docs/phases/`
- **Issue Tracker**: GitHub Issues
- **API Documentation**: Inline code documentation

## License

[Your License Here]

---

**made by harshdeep:/
