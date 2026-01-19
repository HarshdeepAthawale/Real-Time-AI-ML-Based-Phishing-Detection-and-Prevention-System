# Database Schema & Data Models

This directory contains the complete database implementation for the Real-Time AI/ML-Based Phishing Detection and Prevention System.

## Architecture

The system uses three database technologies:

1. **PostgreSQL** - Relational data (users, organizations, threats, detections, ML models, etc.)
2. **MongoDB** - Document storage (email content, URL analysis results, visual analysis)
3. **Redis** - Caching and job queues

## Structure

```
database/
├── schemas/              # PostgreSQL SQL schema files
│   ├── users.sql
│   ├── threats.sql
│   ├── domains.sql
│   ├── ml_models.sql
│   ├── threat_intel.sql
│   ├── emails.sql
│   └── sandbox.sql
├── models/               # TypeORM entities
│   ├── Organization.ts
│   ├── User.ts
│   ├── Threat.ts
│   └── ... (all entities)
├── mongodb/              # MongoDB schemas and connection
│   ├── schemas/
│   │   ├── email-content.ts
│   │   ├── url-analysis.ts
│   │   └── visual-analysis.ts
│   └── connection.ts
├── redis/                # Redis data structures
│   ├── cache-keys.ts
│   ├── queue-keys.ts
│   └── connection.ts
├── migrations/           # Database migrations
│   └── 001-initial-schema.ts
├── connection.ts         # Main database connection manager
├── data-source.ts        # TypeORM data source configuration
├── seed.ts              # Database seeding script
└── migrate.ts           # Migration runner script
```

## Setup

### 1. Install Dependencies

```bash
cd backend/shared
npm install
```

### 2. Configure Environment Variables

Ensure the following environment variables are set:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/phishing_detection
MONGODB_URL=mongodb://localhost:27017/phishing_detection
REDIS_URL=redis://localhost:6379
```

### 3. Run Migrations

```bash
# Build the project first
npm run build

# Run migrations
npm run migration:run

# Or use the migrate script directly
ts-node database/migrate.ts
```

### 4. Seed Database

```bash
npm run seed
```

This will create:
- A default organization (example.com)
- An admin user (admin@example.com / admin123)

## Usage

### PostgreSQL (TypeORM)

```typescript
import { connectPostgreSQL, getPostgreSQL } from '@phishing-detection/shared/database';
import { User } from '@phishing-detection/shared/database/models';

// Connect
await connectPostgreSQL();

// Use repositories
const dataSource = getPostgreSQL();
const userRepository = dataSource.getRepository(User);
const users = await userRepository.find();
```

### MongoDB

```typescript
import { connectMongoDB, getEmailContentCollection } from '@phishing-detection/shared/database';

// Connect
await connectMongoDB();

// Use collections
const emailCollection = getEmailContentCollection();
const email = await emailCollection.findOne({ email_message_id: '...' });
```

### Redis

```typescript
import { connectRedis, getRedis, getQueue } from '@phishing-detection/shared/database';
import { QUEUE_NAMES } from '@phishing-detection/shared/database/redis';

// Connect
await connectRedis();

// Use Redis client
const redis = getRedis();
await redis.set('key', 'value');

// Use queues
const queue = getQueue(QUEUE_NAMES.DETECTION_JOBS);
await queue.add('process-detection', { url: '...' });
```

## Schema Overview

### PostgreSQL Tables

- **organizations** - Organization/tenant management
- **users** - User accounts
- **api_keys** - API key management
- **threats** - Master threat records
- **detections** - Individual detection events
- **threat_indicators** - IOCs associated with threats
- **detection_feedback** - User feedback on detections
- **domains** - Domain information and reputation
- **urls** - URL records and analysis
- **domain_relationships** - Graph relationships between domains
- **ml_models** - ML model metadata
- **model_versions** - Model version tracking
- **training_jobs** - Training job records
- **model_performance** - Model performance metrics
- **threat_intelligence_feeds** - Threat intel feed configuration
- **iocs** - Indicators of Compromise
- **ioc_matches** - IOC detection matches
- **email_messages** - Email metadata
- **email_headers** - Email headers
- **sandbox_analyses** - Sandbox analysis jobs

### MongoDB Collections

- **email_content** - Full email content and NLP analysis
- **url_analysis** - URL graph analysis and GNN results
- **visual_analysis** - Visual analysis and CNN results

### Redis Keys

- **url:reputation:{hash}** - URL reputation cache (1 hour TTL)
- **domain:reputation:{domain}** - Domain reputation cache (6 hours TTL)
- **model:inference:{type}:{hash}** - Model inference cache (24 hours TTL)
- **rate_limit:{key}:{window}** - Rate limiting counters

## Migration Commands

```bash
# Generate a new migration
npm run migration:generate -- -n MigrationName

# Run pending migrations
npm run migration:run

# Revert last migration
npm run migration:revert

# Show migration status
npm run migration:show
```

## Testing

The test setup utilities are available in `test-setup.ts`:

```typescript
import { getTestDatabase, resetTestDatabase, cleanupTestDatabase } from '@phishing-detection/shared/database';

// In your tests
const pool = await getTestDatabase();
await resetTestDatabase(); // Clean slate
// ... run tests
await cleanupTestDatabase(); // Cleanup
```

## Backup Strategy

See [BACKUP_STRATEGY.md](./BACKUP_STRATEGY.md) for comprehensive backup and recovery procedures for all databases.

## Next Steps

After completing Phase 2:
1. ✅ Verify all schemas are created correctly
2. ✅ Test data relationships and foreign keys
3. ✅ Set up database backups (see BACKUP_STRATEGY.md)
4. ✅ Configure connection pooling
5. ➡️ Proceed to Phase 3: NLP Text Analysis Service

## Phase 2 Completion Status

See [PHASE2_COMPLETION.md](./PHASE2_COMPLETION.md) for detailed completion verification.
