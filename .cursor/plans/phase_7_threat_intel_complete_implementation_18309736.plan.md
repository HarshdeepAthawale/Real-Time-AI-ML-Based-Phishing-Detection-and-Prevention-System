# Phase 7: Threat Intelligence Service - Complete Implementation Plan

## Overview

Build a complete end-to-end Threat Intelligence service that integrates with external feeds (MISP, AlienVault OTX), manages IOCs with fast lookup capabilities using Bloom filters, implements scheduled synchronization, and provides a comprehensive REST API.

## Architecture

The service follows the established pattern from `detection-api` service:

- Express.js server with TypeScript
- PostgreSQL via TypeORM (using existing models in `backend/shared/database/models/`)
- Redis via ioredis for Bloom filter persistence and caching
- Scheduled jobs using node-cron
- Structured service layer architecture

## Implementation Components

### 1. Project Structure

Create the following directory structure:

```
backend/core-services/threat-intel/
├── src/
│   ├── index.ts                    # Main application entry
│   ├── config/
│   │   └── index.ts                # Configuration management
│   ├── integrations/
│   │   ├── misp.client.ts          # MISP API client
│   │   ├── otx.client.ts           # AlienVault OTX client
│   │   └── base-feed.client.ts     # Base class for feed clients
│   ├── services/
│   │   ├── ioc-manager.service.ts  # IOC CRUD operations (TypeORM)
│   │   ├── ioc-matcher.service.ts  # Fast IOC matching with Bloom filters
│   │   ├── sync.service.ts         # Feed synchronization logic
│   │   ├── feed-manager.service.ts # Feed management operations
│   │   └── enrichment.service.ts   # IOC enrichment capabilities
│   ├── models/
│   │   └── ioc.model.ts            # IOC domain model (DTO)
│   ├── routes/
│   │   ├── ioc.routes.ts           # IOC check/search endpoints
│   │   ├── feeds.routes.ts         # Feed management endpoints
│   │   └── sync.routes.ts          # Manual sync endpoints
│   ├── middleware/
│   │   ├── error-handler.middleware.ts  # Error handling
│   │   └── validation.middleware.ts     # Request validation
│   ├── utils/
│   │   ├── logger.ts               # Winston logger (already exists)
│   │   ├── bloom-filter.ts         # Bloom filter utilities
│   │   └── normalizers.ts          # IOC value normalization
│   └── jobs/
│       └── sync-scheduler.ts       # Scheduled sync jobs
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── Dockerfile                      # Already exists
├── package.json                    # Update dependencies
├── tsconfig.json                   # TypeScript config
└── README.md                       # Service documentation
```

### 2. Dependencies

**File**: `backend/core-services/threat-intel/package.json`

Add required dependencies:

- `bloom-filters`: ^1.3.0 (Bloom filter implementation)
- `node-cron`: ^3.0.3 (Scheduled jobs)
- `ioredis`: ^5.3.2 (Redis client, replace existing redis)
- `pg`: ^8.11.3 (Already present)
- `typeorm`: ^0.3.17 (TypeORM for database access)
- `zod`: ^3.22.4 (Schema validation)
- Additional Express middleware: `compression`, `helmet`, `cors` (already present)

### 3. Configuration

**File**: `backend/core-services/threat-intel/src/config/index.ts`

Create configuration management:

- Database connection (DATABASE_URL from env)
- Redis connection (REDIS_URL from env)
- MISP configuration (MISP_URL, MISP_API_KEY)
- OTX configuration (OTX_API_KEY)
- Sync schedules (cron expressions)
- Bloom filter settings (size, false positive rate)
- Service port and CORS settings

### 4. Database Integration

Use existing TypeORM models from `backend/shared/database/models/`:

- `IOC` - IOC entity
- `ThreatIntelligenceFeed` - Feed entity
- `IOCMatch` - IOC match tracking

Create DataSource connection or use shared database connection utilities.

### 5. Core Services Implementation

#### 5.1 MISP Client

**File**: `backend/core-services/threat-intel/src/integrations/misp.client.ts`

- Implement MISP REST API client
- Fetch IOCs with date filtering
- Publish IOCs to MISP
- Map MISP attribute types to IOC types
- Map MISP threat levels to severity levels
- Error handling and retry logic

#### 5.2 OTX Client

**File**: `backend/core-services/threat-intel/src/integrations/otx.client.ts`

- Implement AlienVault OTX API client
- Fetch subscribed pulses with date filtering
- Extract indicators from pulses
- Map OTX TLP levels to severity
- Handle pagination
- Error handling

#### 5.3 IOC Manager Service

**File**: `backend/core-services/threat-intel/src/services/ioc-manager.service.ts`

- CRUD operations using TypeORM
- Bulk insert with conflict handling (ON CONFLICT UPDATE)
- Hash-based IOC value indexing
- IOC search and filtering
- Statistics and reporting

#### 5.4 IOC Matcher Service

**File**: `backend/core-services/threat-intel/src/services/ioc-matcher.service.ts`

- Bloom filter initialization per IOC type
- Fast negative lookup (definitely not in DB)
- Database fallback for positive matches
- Bloom filter persistence in Redis
- Bulk bloom filter updates
- Automatic bloom filter rebuild when size limit reached

#### 5.5 Sync Service

**File**: `backend/core-services/threat-intel/src/services/sync.service.ts`

- Orchestrate feed synchronization
- Track last sync time per feed
- Incremental sync (only fetch new IOCs since last sync)
- Bulk IOC insertion
- Bloom filter updates after sync
- Error handling and status tracking
- Update feed sync status in database

#### 5.6 Feed Manager Service

**File**: `backend/core-services/threat-intel/src/services/feed-manager.service.ts`

- CRUD operations for feeds
- Feed activation/deactivation
- Feed health monitoring
- Reliability scoring

### 6. API Routes

#### 6.1 IOC Routes

**File**: `backend/core-services/threat-intel/src/routes/ioc.routes.ts`

- `POST /api/v1/ioc/check` - Check if IOC exists
- `POST /api/v1/ioc/bulk-check` - Bulk IOC checking
- `GET /api/v1/ioc/search` - Search IOCs
- `POST /api/v1/ioc/report` - Report new IOC
- `GET /api/v1/ioc/stats` - IOC statistics

#### 6.2 Feed Routes

**File**: `backend/core-services/threat-intel/src/routes/feeds.routes.ts`

- `GET /api/v1/feeds` - List all feeds
- `GET /api/v1/feeds/:id` - Get feed details
- `POST /api/v1/feeds` - Create new feed
- `PUT /api/v1/feeds/:id` - Update feed
- `DELETE /api/v1/feeds/:id` - Delete feed
- `POST /api/v1/feeds/:id/toggle` - Enable/disable feed

#### 6.3 Sync Routes

**File**: `backend/core-services/threat-intel/src/routes/sync.routes.ts`

- `POST /api/v1/sync/all` - Trigger sync for all feeds
- `POST /api/v1/sync/:feedId` - Trigger sync for specific feed
- `GET /api/v1/sync/status` - Get sync status

### 7. Scheduled Jobs

**File**: `backend/core-services/threat-intel/src/jobs/sync-scheduler.ts`

- Initialize cron jobs on service startup
- Configurable sync intervals per feed
- Handle job failures gracefully
- Log sync activities
- Update last sync timestamps

### 8. Utilities

#### 8.1 IOC Normalizers

**File**: `backend/core-services/threat-intel/src/utils/normalizers.ts`

- Normalize URLs (remove protocol, lowercase, etc.)
- Normalize domains (lowercase, remove www)
- Normalize IP addresses
- Normalize hashes (uppercase, remove spaces)
- Consistent formatting for lookups

#### 8.2 Bloom Filter Utilities

**File**: `backend/core-services/threat-intel/src/utils/bloom-filter.ts`

- Bloom filter factory functions
- Serialization/deserialization helpers
- Size estimation utilities
- Merge operations

### 9. Main Application Entry

**File**: `backend/core-services/threat-intel/src/index.ts`

- Express app setup
- Middleware configuration (helmet, cors, compression)
- Database connection (TypeORM)
- Redis connection (ioredis)
- Service initialization
- Route registration
- Scheduled jobs startup
- Health check endpoint
- Graceful shutdown handling

### 10. Integration Points

#### 10.1 Detection API Integration

- Detection API should call threat-intel service for IOC matching
- Update `orchestrator.service.ts` in detection-api to check IOCs
- WebSocket events for IOC matches

#### 10.2 Database Schema

- Use existing `threat_intel.sql` schema
- Ensure migrations are run
- Verify indexes are created

#### 10.3 Docker Configuration

- Update `backend/docker-compose.yml` with threat-intel environment variables
- Add dependencies (postgres, redis)

### 11. Testing

Create comprehensive tests:

- Unit tests for each service
- Integration tests for API endpoints
- Integration tests for feed clients (with mocks)
- Bloom filter accuracy tests
- Sync service tests

### 12. Documentation

**File**: `backend/core-services/threat-intel/README.md`

- Service overview
- API documentation
- Configuration guide
- Feed setup instructions (MISP, OTX)
- Development guide
- Deployment notes

## Implementation Order

1. **Phase 1: Foundation**

   - Configuration setup
   - Database connection (TypeORM)
   - Redis connection
   - Basic Express server with health check
   - Logger setup

2. **Phase 2: Core Services**

   - IOC Manager Service
   - IOC Matcher Service (without Bloom filters initially)
   - Basic IOC CRUD operations

3. **Phase 3: Feed Clients**

   - Base feed client interface
   - MISP client implementation
   - OTX client implementation
   - Test with mock responses

4. **Phase 4: Bloom Filters**

   - Bloom filter utilities
   - IOC Matcher integration
   - Redis persistence
   - Bloom filter rebuilding logic

5. **Phase 5: Synchronization**

   - Sync Service implementation
   - Feed Manager Service
   - Last sync time tracking
   - Error handling and retries

6. **Phase 6: API Endpoints**

   - IOC routes
   - Feed routes
   - Sync routes
   - Request validation
   - Error handling middleware

7. **Phase 7: Scheduled Jobs**

   - Sync scheduler implementation
   - Cron job configuration
   - Job monitoring

8. **Phase 8: Integration & Testing**

   - Integration with detection-api
   - End-to-end testing
   - Performance testing
   - Documentation

## Key Considerations

1. **Performance**: Bloom filters enable sub-millisecond negative lookups
2. **Scalability**: Bulk operations for large IOC imports
3. **Reliability**: Feed sync error handling and retries
4. **Security**: API key encryption for feed credentials
5. **Monitoring**: Comprehensive logging and status tracking
6. **Maintainability**: Follow existing service patterns

## Environment Variables

Required environment variables:

- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `MISP_URL` - MISP instance URL (optional)
- `MISP_API_KEY` - MISP API key (optional)
- `OTX_API_KEY` - AlienVault OTX API key (optional)
- `PORT` - Service port (default: 3002)
- `NODE_ENV` - Environment (development/production)
- `LOG_LEVEL` - Logging level

## Deliverables Checklist

- [ ] All service files implemented
- [ ] MISP client functional
- [ ] OTX client functional
- [ ] IOC manager with TypeORM
- [ ] IOC matcher with Bloom filters
- [ ] Sync service with incremental updates
- [ ] Scheduled sync jobs configured
- [ ] REST API endpoints complete
- [ ] Error handling and validation
- [ ] Health check endpoint
- [ ] Tests written (unit + integration)
- [ ] Documentation complete
- [ ] Docker configuration updated
- [ ] Integration with detection-api