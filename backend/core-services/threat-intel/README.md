# Threat Intelligence Service

A comprehensive threat intelligence service that integrates with external feeds (MISP, AlienVault OTX), manages Indicators of Compromise (IOCs) with fast lookup capabilities using Bloom filters, implements scheduled synchronization, and provides a REST API.

## Features

- **Feed Integration**: Support for MISP and AlienVault OTX threat intelligence feeds
- **IOC Management**: CRUD operations for Indicators of Compromise
- **Fast Lookups**: Bloom filter-based negative lookups for sub-millisecond performance
- **Scheduled Sync**: Automatic synchronization with configurable intervals
- **REST API**: Complete RESTful API for IOC checking, feed management, and synchronization
- **Incremental Sync**: Only fetch new IOCs since last synchronization
- **Reliability Tracking**: Monitor feed health and reliability scores

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│      Threat Intelligence Service                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   MISP       │  │   OTX        │  │   Custom     │  │
│  │   Connector  │  │   Connector  │  │   Feeds      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   IOC        │  │   Bloom      │  │   Sync       │  │
│  │   Manager    │  │   Filter     │  │   Scheduler  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## API Endpoints

### IOC Management

- `POST /api/v1/ioc/check` - Check if an IOC exists
- `POST /api/v1/ioc/bulk-check` - Bulk IOC checking (up to 100 IOCs)
- `GET /api/v1/ioc/search` - Search IOCs with filters
- `POST /api/v1/ioc/report` - Report a new IOC
- `GET /api/v1/ioc/stats` - Get IOC statistics

### Feed Management

- `GET /api/v1/feeds` - List all feeds
- `GET /api/v1/feeds/:id` - Get feed details
- `POST /api/v1/feeds` - Create new feed
- `PUT /api/v1/feeds/:id` - Update feed
- `DELETE /api/v1/feeds/:id` - Delete feed
- `POST /api/v1/feeds/:id/toggle` - Enable/disable feed

### Synchronization

- `POST /api/v1/sync/all` - Trigger sync for all feeds
- `POST /api/v1/sync/:feedId` - Trigger sync for specific feed
- `GET /api/v1/sync/status` - Get sync status

### Health

- `GET /health` - Health check endpoint

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Service port | `3002` |
| `NODE_ENV` | Environment | `development` |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `MISP_URL` | MISP instance URL | Optional |
| `MISP_API_KEY` | MISP API key | Optional |
| `OTX_API_KEY` | AlienVault OTX API key | Optional |
| `LOG_LEVEL` | Logging level | `info` |
| `SYNC_INTERVAL_MINUTES` | Default sync interval | `60` |
| `BLOOM_FILTER_SIZE` | Bloom filter size | `1000000` |
| `BLOOM_FILTER_FALSE_POSITIVE_RATE` | False positive rate | `0.01` |

## Setup

### Prerequisites

- Node.js 20+
- PostgreSQL 15+
- Redis 7+
- (Optional) MISP instance with API access
- (Optional) AlienVault OTX API key

### Installation

```bash
cd backend/core-services/threat-intel
npm install
npm run build
```

### Database Setup

The service uses the existing database schema from `backend/shared/database/schemas/threat_intel.sql`. Ensure migrations have been run:

```bash
# From backend/shared/database
npm run migrate
```

### Running

Development:
```bash
npm run dev
```

Production:
```bash
npm run build
npm start
```

### Docker

```bash
docker-compose up threat-intel
```

## Usage

### Checking an IOC

```bash
curl -X POST http://localhost:3002/api/v1/ioc/check \
  -H "Content-Type: application/json" \
  -d '{
    "iocType": "domain",
    "iocValue": "malicious.example.com"
  }'
```

### Bulk Check IOCs

```bash
curl -X POST http://localhost:3002/api/v1/ioc/bulk-check \
  -H "Content-Type: application/json" \
  -d '{
    "iocs": [
      {"iocType": "domain", "iocValue": "example1.com"},
      {"iocType": "ip", "iocValue": "192.168.1.1"}
    ]
  }'
```

### Creating a Feed

```bash
curl -X POST http://localhost:3002/api/v1/feeds \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MISP Feed",
    "feedType": "misp",
    "apiEndpoint": "https://misp.example.com",
    "apiKeyEncrypted": "your-api-key",
    "syncIntervalMinutes": 60,
    "isActive": true
  }'
```

### Triggering Sync

```bash
# Sync all feeds
curl -X POST http://localhost:3002/api/v1/sync/all

# Sync specific feed
curl -X POST http://localhost:3002/api/v1/sync/{feedId}
```

## Feed Setup

### MISP Setup

1. Get your MISP API key from your MISP instance
2. Create a feed in the service:
   ```bash
   curl -X POST http://localhost:3002/api/v1/feeds \
     -H "Content-Type: application/json" \
     -d '{
       "name": "MISP Production",
       "feedType": "misp",
       "apiEndpoint": "https://your-misp-instance.com",
       "apiKeyEncrypted": "your-misp-api-key",
       "syncIntervalMinutes": 60,
       "isActive": true
     }'
   ```

### OTX Setup

1. Get your OTX API key from [AlienVault OTX](https://otx.alienvault.com/api)
2. Create a feed in the service:
   ```bash
   curl -X POST http://localhost:3002/api/v1/feeds \
     -H "Content-Type: application/json" \
     -d '{
       "name": "OTX Feed",
       "feedType": "otx",
       "apiKeyEncrypted": "your-otx-api-key",
       "syncIntervalMinutes": 60,
       "isActive": true
     }'
   ```

## Bloom Filters

The service uses Bloom filters for fast negative lookups:

- **Performance**: Sub-millisecond lookups for IOCs not in the database
- **Storage**: Persisted in Redis with configurable TTL
- **Rebuild**: Automatic rebuild when size limits are reached
- **Types**: Separate bloom filters for each IOC type (url, domain, ip, etc.)

## Development

### Project Structure

```
src/
├── index.ts                    # Main application entry
├── config/
│   └── index.ts                # Configuration
├── integrations/
│   ├── misp.client.ts          # MISP API client
│   ├── otx.client.ts           # OTX API client
│   └── base-feed.client.ts     # Base feed client
├── services/
│   ├── ioc-manager.service.ts  # IOC CRUD operations
│   ├── ioc-matcher.service.ts  # IOC matching with bloom filters
│   ├── sync.service.ts         # Feed synchronization
│   ├── feed-manager.service.ts # Feed management
│   ├── database.service.ts     # Database connection
│   └── redis.service.ts        # Redis connection
├── routes/
│   ├── ioc.routes.ts           # IOC endpoints
│   ├── feeds.routes.ts         # Feed endpoints
│   └── sync.routes.ts          # Sync endpoints
├── middleware/
│   ├── error-handler.middleware.ts
│   └── validation.middleware.ts
├── models/
│   └── ioc.model.ts            # IOC domain models
├── utils/
│   ├── logger.ts
│   ├── bloom-filter.ts
│   └── normalizers.ts
└── jobs/
    └── sync-scheduler.ts       # Scheduled sync jobs
```

### Testing

```bash
# Run all tests
npm test

# Run unit tests
npm run test:unit

# Run integration tests
npm run test:integration
```

## Performance

- **IOC Lookups**: < 1ms for negative lookups (not in database)
- **Database Queries**: Optimized with indexes on `(ioc_type, ioc_value_hash)`
- **Sync Performance**: Handles thousands of IOCs per sync
- **Bloom Filter**: Memory-efficient probabilistic data structure

## Security

- API keys stored encrypted in database
- Input validation using Zod schemas
- Helmet.js for HTTP security headers
- CORS configuration
- Rate limiting (via API gateway)

## Monitoring

- Health check endpoint: `/health`
- Comprehensive logging via Winston
- Feed reliability scoring
- Sync status tracking

## Integration with Detection API

The Threat Intelligence service integrates with the Detection API:

1. Detection API calls IOC check endpoints during analysis
2. IOC matches are tracked in the database
3. WebSocket events can be sent for IOC matches

## Troubleshooting

### Bloom Filter Not Working

- Check Redis connection
- Verify bloom filter initialization in logs
- Rebuild bloom filters if needed

### Sync Failures

- Check feed configuration (API keys, endpoints)
- Verify network connectivity to feed sources
- Check logs for specific error messages
- Review feed sync status via API

### Database Connection Issues

- Verify `DATABASE_URL` environment variable
- Check PostgreSQL is running and accessible
- Ensure database migrations have been run

## License

ISC
