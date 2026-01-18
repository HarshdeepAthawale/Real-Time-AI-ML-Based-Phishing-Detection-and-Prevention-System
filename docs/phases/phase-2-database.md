# Phase 2: Database Schema & Data Models

## Objective
Design and implement comprehensive database schemas for all system entities across PostgreSQL (relational), MongoDB (document), and Redis (cache/queue).

## Prerequisites
- Phase 1 infrastructure completed
- PostgreSQL, MongoDB, and Redis instances running
- Database migration tool selected (TypeORM, Prisma, or raw SQL)

## Database Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    PostgreSQL                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Users &    │  │   Threats &  │  │   ML Models  │ │
│  │ Organizations│  │  Detections  │  │  & Training  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                      MongoDB                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Email      │  │   URL/Graph  │  │   Visual     │ │
│  │   Content    │  │   Analysis   │  │   Analysis   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                       Redis                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Cache      │  │   Queues     │  │   Sessions   │ │
│  │   (URLs,     │  │   (Jobs,     │  │   & Locks    │ │
│  │   Domains)   │  │   Events)    │  │              │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## PostgreSQL Schema Design

### 1. Users & Organizations

**File**: `backend/shared/database/schemas/users.sql`

```sql
-- Organizations table
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE,
    plan VARCHAR(50) DEFAULT 'free', -- free, pro, enterprise
    max_users INTEGER DEFAULT 10,
    max_api_calls_per_day INTEGER DEFAULT 10000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'user', -- admin, user, viewer
    is_active BOOLEAN DEFAULT true,
    last_login_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

-- API Keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_prefix VARCHAR(20) NOT NULL, -- First 8 chars for identification
    name VARCHAR(255) NOT NULL,
    permissions JSONB DEFAULT '{}',
    rate_limit_per_minute INTEGER DEFAULT 100,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP
);

CREATE INDEX idx_users_organization ON users(organization_id);
CREATE INDEX idx_api_keys_organization ON api_keys(organization_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
```

### 2. Threats & Detections

**File**: `backend/shared/database/schemas/threats.sql`

```sql
-- Threats table (master threat records)
CREATE TABLE threats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    threat_type VARCHAR(50) NOT NULL, -- email_phishing, url_spoofing, domain_hijacking, ai_generated
    severity VARCHAR(20) NOT NULL, -- critical, high, medium, low
    status VARCHAR(20) DEFAULT 'detected', -- detected, blocked, resolved, false_positive
    confidence_score DECIMAL(5,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 100),
    source VARCHAR(50), -- email, url, domain, file
    source_value TEXT, -- The actual email/URL/domain that triggered detection
    title VARCHAR(500),
    description TEXT,
    metadata JSONB DEFAULT '{}', -- Additional context
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detections table (individual detection events)
CREATE TABLE detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    threat_id UUID REFERENCES threats(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    detection_type VARCHAR(50) NOT NULL, -- nlp, url, visual, ioc, ensemble
    model_version VARCHAR(50),
    input_data JSONB NOT NULL, -- What was analyzed
    analysis_result JSONB NOT NULL, -- ML model output
    confidence_score DECIMAL(5,2) NOT NULL,
    processing_time_ms INTEGER,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Threat Indicators table (IOCs associated with threats)
CREATE TABLE threat_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    threat_id UUID REFERENCES threats(id) ON DELETE CASCADE,
    indicator_type VARCHAR(50) NOT NULL, -- url, domain, ip, email, hash, filename
    indicator_value TEXT NOT NULL,
    source VARCHAR(50), -- detection, threat_intel, user_report
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User feedback on detections
CREATE TABLE detection_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    feedback_type VARCHAR(20) NOT NULL, -- true_positive, false_positive, false_negative
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_threats_organization ON threats(organization_id);
CREATE INDEX idx_threats_detected_at ON threats(detected_at DESC);
CREATE INDEX idx_threats_status ON threats(status);
CREATE INDEX idx_detections_threat ON detections(threat_id);
CREATE INDEX idx_detections_organization ON detections(organization_id);
CREATE INDEX idx_detections_detected_at ON detections(detected_at DESC);
CREATE INDEX idx_threat_indicators_threat ON threat_indicators(threat_id);
CREATE INDEX idx_threat_indicators_value ON threat_indicators(indicator_type, indicator_value);
CREATE INDEX idx_detection_feedback_detection ON detection_feedback(detection_id);
```

### 3. Domains & URLs

**File**: `backend/shared/database/schemas/domains.sql`

```sql
-- Domains table
CREATE TABLE domains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain VARCHAR(255) UNIQUE NOT NULL,
    tld VARCHAR(50),
    subdomain VARCHAR(255),
    registered_domain VARCHAR(255), -- Base domain without subdomain
    reputation_score DECIMAL(5,2) DEFAULT 50 CHECK (reputation_score >= 0 AND reputation_score <= 100),
    is_malicious BOOLEAN DEFAULT false,
    is_suspicious BOOLEAN DEFAULT false,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_analyzed_at TIMESTAMP,
    whois_data JSONB,
    dns_records JSONB,
    ssl_certificate_data JSONB,
    analysis_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- URLs table
CREATE TABLE urls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id UUID REFERENCES domains(id) ON DELETE CASCADE,
    full_url TEXT NOT NULL,
    url_hash VARCHAR(64) UNIQUE NOT NULL, -- SHA-256 hash
    scheme VARCHAR(10), -- http, https
    path TEXT,
    query_params JSONB,
    fragment TEXT,
    redirect_chain JSONB, -- Array of redirect URLs
    redirect_count INTEGER DEFAULT 0,
    is_malicious BOOLEAN DEFAULT false,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_analyzed_at TIMESTAMP,
    analysis_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Domain relationships (for graph analysis)
CREATE TABLE domain_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_domain_id UUID REFERENCES domains(id) ON DELETE CASCADE,
    target_domain_id UUID REFERENCES domains(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL, -- redirects_to, shares_ip, shares_registrar, similar_name
    strength DECIMAL(5,2) DEFAULT 1.0, -- Relationship strength score
    metadata JSONB DEFAULT '{}',
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_domain_id, target_domain_id, relationship_type)
);

CREATE INDEX idx_domains_domain ON domains(domain);
CREATE INDEX idx_domains_registered_domain ON domains(registered_domain);
CREATE INDEX idx_domains_reputation ON domains(reputation_score);
CREATE INDEX idx_urls_domain ON urls(domain_id);
CREATE INDEX idx_urls_hash ON urls(url_hash);
CREATE INDEX idx_urls_malicious ON urls(is_malicious);
CREATE INDEX idx_domain_relationships_source ON domain_relationships(source_domain_id);
CREATE INDEX idx_domain_relationships_target ON domain_relationships(target_domain_id);
```

### 4. ML Models & Training

**File**: `backend/shared/database/schemas/ml_models.sql`

```sql
-- ML Models table
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(50) NOT NULL, -- nlp, url_gnn, visual_cnn, adversarial
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    model_path_s3 TEXT, -- S3 path to model file
    model_size_bytes BIGINT,
    framework VARCHAR(50), -- pytorch, tensorflow, onnx
    input_schema JSONB, -- Expected input format
    output_schema JSONB, -- Expected output format
    metrics JSONB DEFAULT '{}', -- accuracy, precision, recall, f1
    training_config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT false,
    deployed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_type, version)
);

-- Model versions (tracking all versions)
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_models(id) ON DELETE CASCADE,
    version VARCHAR(50) NOT NULL,
    model_path_s3 TEXT,
    metrics JSONB DEFAULT '{}',
    training_job_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training jobs
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    training_config JSONB NOT NULL,
    dataset_path_s3 TEXT,
    dataset_size INTEGER,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    metrics JSONB DEFAULT '{}',
    error_message TEXT,
    logs_s3_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance monitoring
CREATE TABLE model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES ml_models(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    avg_inference_time_ms DECIMAL(10,2),
    accuracy DECIMAL(5,2),
    precision DECIMAL(5,2),
    recall DECIMAL(5,2),
    f1_score DECIMAL(5,2),
    false_positive_rate DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, date)
);

CREATE INDEX idx_ml_models_type ON ml_models(model_type);
CREATE INDEX idx_ml_models_active ON ml_models(is_active);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_model_type ON training_jobs(model_type);
CREATE INDEX idx_model_performance_model ON model_performance(model_id);
CREATE INDEX idx_model_performance_date ON model_performance(date DESC);
```

### 5. Threat Intelligence

**File**: `backend/shared/database/schemas/threat_intel.sql`

```sql
-- Threat intelligence feeds
CREATE TABLE threat_intelligence_feeds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    feed_type VARCHAR(50) NOT NULL, -- misp, otx, custom, user_submitted
    api_endpoint TEXT,
    api_key_encrypted TEXT,
    sync_interval_minutes INTEGER DEFAULT 60,
    last_sync_at TIMESTAMP,
    last_sync_status VARCHAR(20), -- success, failed, partial
    last_sync_error TEXT,
    is_active BOOLEAN DEFAULT true,
    reliability_score DECIMAL(5,2) DEFAULT 50,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- IOCs (Indicators of Compromise)
CREATE TABLE iocs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feed_id UUID REFERENCES threat_intelligence_feeds(id) ON DELETE SET NULL,
    ioc_type VARCHAR(50) NOT NULL, -- url, domain, ip, email, hash_md5, hash_sha1, hash_sha256, filename
    ioc_value TEXT NOT NULL,
    ioc_value_hash VARCHAR(64), -- For fast lookups
    threat_type VARCHAR(100),
    severity VARCHAR(20),
    confidence DECIMAL(5,2),
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    source_reports INTEGER DEFAULT 1,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ioc_type, ioc_value_hash)
);

-- IOC matches (when our system detects an IOC)
CREATE TABLE ioc_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ioc_id UUID REFERENCES iocs(id) ON DELETE CASCADE,
    detection_id UUID REFERENCES detections(id) ON DELETE CASCADE,
    matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_iocs_type_value ON iocs(ioc_type, ioc_value_hash);
CREATE INDEX idx_iocs_feed ON iocs(feed_id);
CREATE INDEX idx_iocs_severity ON iocs(severity);
CREATE INDEX idx_ioc_matches_ioc ON ioc_matches(ioc_id);
CREATE INDEX idx_ioc_matches_detection ON ioc_matches(detection_id);
```

### 6. Email Messages

**File**: `backend/shared/database/schemas/emails.sql`

```sql
-- Email messages (metadata only, full content in MongoDB)
CREATE TABLE email_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    message_id VARCHAR(500) UNIQUE, -- Email Message-ID header
    from_email VARCHAR(255),
    to_emails TEXT[], -- Array of recipient emails
    subject TEXT,
    received_at TIMESTAMP,
    analyzed_at TIMESTAMP,
    threat_id UUID REFERENCES threats(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Email headers (parsed headers)
CREATE TABLE email_headers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email_message_id UUID REFERENCES email_messages(id) ON DELETE CASCADE,
    header_name VARCHAR(255) NOT NULL,
    header_value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_email_messages_org ON email_messages(organization_id);
CREATE INDEX idx_email_messages_message_id ON email_messages(message_id);
CREATE INDEX idx_email_messages_received_at ON email_messages(received_at DESC);
CREATE INDEX idx_email_headers_email ON email_headers(email_message_id);
```

### 7. Sandbox Analyses

**File**: `backend/shared/database/schemas/sandbox.sql`

```sql
-- Sandbox analysis jobs
CREATE TABLE sandbox_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL, -- url, file
    target_url TEXT,
    target_file_hash VARCHAR(64),
    sandbox_provider VARCHAR(50), -- cuckoo, anyrun, custom
    sandbox_job_id VARCHAR(255),
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    result_data JSONB,
    threat_id UUID REFERENCES threats(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sandbox_analyses_org ON sandbox_analyses(organization_id);
CREATE INDEX idx_sandbox_analyses_status ON sandbox_analyses(status);
CREATE INDEX idx_sandbox_analyses_submitted_at ON sandbox_analyses(submitted_at DESC);
```

## MongoDB Collections Design

### 1. Email Content Collection

**File**: `backend/shared/database/mongodb/schemas/email-content.ts`

```typescript
interface EmailContent {
  _id: ObjectId;
  email_message_id: string; // Reference to PostgreSQL email_messages.id
  body_text: string;
  body_html: string;
  attachments?: Array<{
    filename: string;
    content_type: string;
    size: number;
    hash: string;
  }>;
  nlp_analysis?: {
    embeddings: number[];
    sentiment: string;
    urgency_score: number;
    ai_generated_probability: number;
    features: Record<string, any>;
  };
  created_at: Date;
  updated_at: Date;
}

// Indexes
db.email_content.createIndex({ email_message_id: 1 }, { unique: true });
db.email_content.createIndex({ "nlp_analysis.ai_generated_probability": -1 });
```

### 2. URL Analysis Results Collection

**File**: `backend/shared/database/mongodb/schemas/url-analysis.ts`

```typescript
interface URLAnalysis {
  _id: ObjectId;
  url_id: string; // Reference to PostgreSQL urls.id
  domain_id: string; // Reference to PostgreSQL domains.id
  graph_analysis?: {
    node_embeddings: number[];
    cluster_id: string;
    relationships: Array<{
      related_domain_id: string;
      relationship_type: string;
      strength: number;
    }>;
  };
  gnn_analysis?: {
    malicious_probability: number;
    anomaly_score: number;
    features: Record<string, any>;
  };
  redirect_chain_analysis?: {
    hops: Array<{
      url: string;
      status_code: number;
      redirect_type: string;
    }>;
    suspicious_patterns: string[];
  };
  created_at: Date;
  updated_at: Date;
}

// Indexes
db.url_analysis.createIndex({ url_id: 1 }, { unique: true });
db.url_analysis.createIndex({ domain_id: 1 });
db.url_analysis.createIndex({ "gnn_analysis.malicious_probability": -1 });
```

### 3. Visual Analysis Results Collection

**File**: `backend/shared/database/mongodb/schemas/visual-analysis.ts`

```typescript
interface VisualAnalysis {
  _id: ObjectId;
  url_id: string;
  screenshot_s3_path: string;
  dom_structure: {
    tree_hash: string;
    element_count: number;
    form_fields: Array<{
      type: string;
      name: string;
      placeholder?: string;
    }>;
    links: Array<{
      href: string;
      text: string;
    }>;
  };
  cnn_analysis?: {
    brand_impersonation_score: number;
    visual_similarity_scores: Array<{
      legitimate_domain: string;
      similarity: number;
    }>;
    features: Record<string, any>;
  };
  created_at: Date;
  updated_at: Date;
}

// Indexes
db.visual_analysis.createIndex({ url_id: 1 }, { unique: true });
db.visual_analysis.createIndex({ "cnn_analysis.brand_impersonation_score": -1 });
```

## Redis Data Structures

### 1. Cache Keys

**File**: `backend/shared/database/redis/cache-keys.ts`

```typescript
// URL reputation cache
// Key: `url:reputation:{url_hash}`
// Value: JSON string with reputation score and metadata
// TTL: 1 hour

// Domain reputation cache
// Key: `domain:reputation:{domain}`
// Value: JSON string with reputation score
// TTL: 6 hours

// IOC lookup cache (Bloom filter for fast negative lookups)
// Key: `ioc:bloom:{ioc_type}`
// Use RedisBloom module

// Model inference cache
// Key: `model:inference:{model_type}:{input_hash}`
// Value: JSON string with model output
// TTL: 24 hours
```

### 2. Queue Keys

**File**: `backend/shared/database/redis/queue-keys.ts`

```typescript
// Job queues (using BullMQ)
// Queue: `detection-jobs` - For async detection processing
// Queue: `sandbox-jobs` - For sandbox submissions
// Queue: `training-jobs` - For model training
// Queue: `threat-intel-sync` - For threat intel feed synchronization

// Rate limiting
// Key: `rate_limit:{api_key}:{window}` (e.g., `rate_limit:abc123:minute`)
// Value: Counter
// TTL: Window duration
```

## Migration System

### Using TypeORM Migrations

**File**: `backend/shared/database/migrations/001-initial-schema.ts`

```typescript
import { MigrationInterface, QueryRunner } from 'typeorm';

export class InitialSchema1234567890 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    // Execute SQL from schema files
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    // Rollback logic
  }
}
```

## Data Models (TypeScript)

### TypeORM Entities

**File**: `backend/shared/database/models/Threat.ts`

```typescript
import { Entity, Column, PrimaryGeneratedColumn, ManyToOne, JoinColumn } from 'typeorm';
import { Organization } from './Organization';

@Entity('threats')
export class Threat {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ManyToOne(() => Organization)
  @JoinColumn({ name: 'organization_id' })
  organization: Organization;

  @Column()
  threat_type: string;

  @Column()
  severity: string;

  @Column({ type: 'decimal', precision: 5, scale: 2 })
  confidence_score: number;

  // ... other fields
}
```

## Deliverables Checklist

- [ ] PostgreSQL schema files created
- [ ] All tables created with proper indexes
- [ ] Foreign key constraints defined
- [ ] MongoDB collections designed
- [ ] MongoDB indexes created
- [ ] Redis data structures documented
- [ ] TypeORM/Prisma models created
- [ ] Migration system set up
- [ ] Database seeding scripts
- [ ] Connection pooling configured
- [ ] Backup strategy documented

## Testing

### Database Tests
- All tables can be created
- Foreign keys work correctly
- Indexes improve query performance
- Migrations can be applied and rolled back
- Seed data loads correctly

### Performance Tests
- Query performance meets requirements (<10ms for simple queries)
- Index usage is optimal
- Connection pooling works correctly
- MongoDB queries are efficient

## Next Steps

After completing Phase 2:
1. Verify all schemas are created
2. Test data relationships
3. Set up database backups
4. Configure connection pooling
5. Proceed to Phase 3: NLP Text Analysis Service
