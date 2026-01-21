---
name: Remove Mocks Use Real Services
overview: Remove all jest.mock() calls and refactor tests to use real services (PostgreSQL, Redis, external APIs) via Docker Compose test infrastructure. All tests must pass with real service connections.
todos:
  - id: "1"
    content: Create docker-compose.test.yml with test database, Redis, and MongoDB services
    status: completed
  - id: "2"
    content: Create test helper utilities (test-db.ts, test-redis.ts, test-services.ts)
    status: completed
  - id: "3"
    content: Create test fixtures for API keys and database seeders
    status: completed
  - id: "4"
    content: Update tests/setup.ts to initialize real database and Redis connections
    status: completed
  - id: "5"
    content: Refactor URL checker service tests to use real services
    status: completed
  - id: "6"
    content: Refactor email scanner service tests to use real services
    status: completed
  - id: "7"
    content: Refactor cache service tests to use real Redis
    status: completed
  - id: "8"
    content: Refactor email client service tests (handle IMAP requirement)
    status: completed
  - id: "9"
    content: Refactor extension auth middleware tests to use real database
    status: completed
  - id: "10"
    content: Refactor rate limit middleware tests to use real Redis
    status: completed
  - id: "11"
    content: Refactor URL check route integration tests to use real services
    status: completed
  - id: "12"
    content: Refactor email scan route integration tests to use real services
    status: completed
  - id: "13"
    content: Refactor report route integration tests to use real threat-intel service
    status: completed
  - id: "14"
    content: Refactor email client route integration tests
    status: completed
  - id: "15"
    content: Update package.json with test scripts for Docker infrastructure
    status: completed
  - id: "16"
    content: Create .env.test file with test environment variables
    status: completed
  - id: "17"
    content: Run all tests and fix any issues to ensure 100% pass rate
    status: completed
---

# Remove All Mocks and Use Real Services

## Overview

Refactor all tests to use real services instead of mocks. Tests will connect to real PostgreSQL, Redis, and external APIs (detection-api, threat-intel) running via Docker Compose.

## Architecture Changes

### Test Infrastructure Setup

1. **Create `docker-compose.test.yml`**

                                                                                                - Separate test database (`phishing_detection_test`)
                                                                                                - Test Redis instance (separate port)
                                                                                                - Test MongoDB instance (separate port)
                                                                                                - Services run on different ports to avoid conflicts

2. **Test Environment Configuration**

                                                                                                - Environment variables for test connections
                                                                                                - Test database URL: `postgresql://postgres:postgres@localhost:5433/phishing_detection_test`
                                                                                                - Test Redis URL: `redis://localhost:6380`
                                                                                                - Test MongoDB URL: `mongodb://localhost:27018/phishing_detection_test`

### Test Setup Refactoring

**File**: `backend/core-services/extension-api/tests/setup.ts`

- Remove logger mock
- Add real database connection initialization
- Add real Redis connection initialization
- Add database migration/seed before tests
- Add cleanup after all tests
- Set up test environment variables

**File**: `backend/core-services/extension-api/tests/helpers/test-db.ts` (new)

- Database connection helpers using shared utilities
- Test database migration runner
- Test data seeders
- Cleanup utilities (truncate tables between tests)

**File**: `backend/core-services/extension-api/tests/helpers/test-redis.ts` (new)

- Redis connection helpers
- Redis cleanup utilities (flush between tests)

**File**: `backend/core-services/extension-api/tests/helpers/test-services.ts` (new)

- Helpers to check if external services are available
- Service health check utilities
- Skip tests if services unavailable

## Unit Tests Refactoring

### 1. URL Checker Service Tests

**File**: `backend/core-services/extension-api/tests/unit/services/url-checker.service.test.ts`

- Remove: `jest.mock('axios')`, `jest.mock('cache.service')`, `jest.mock('privacy-filter.service')`
- Use: Real `CacheService` instance with test Redis
- Use: Real `PrivacyFilterService` instance
- Use: Real axios calls to detection-api (must be running)
- Add: BeforeEach to initialize real services
- Add: AfterEach to cleanup Redis cache

### 2. Email Scanner Service Tests

**File**: `backend/core-services/extension-api/tests/unit/services/email-scanner.service.test.ts`

- Remove: All mocks
- Use: Real `PrivacyFilterService` and `URLCheckerService`
- Use: Real axios calls to detection-api
- Add: Service initialization and cleanup

### 3. Privacy Filter Service Tests

**File**: `backend/core-services/extension-api/tests/unit/services/privacy-filter.service.test.ts`

- No changes needed (already uses real service, no mocks)

### 4. Cache Service Tests

**File**: `backend/core-services/extension-api/tests/unit/services/cache.service.test.ts`

- Remove: `jest.mock('ioredis')`
- Use: Real Redis connection (test instance)
- Add: BeforeEach to connect to test Redis
- Add: AfterEach to flush Redis

### 5. Email Client Service Tests

**File**: `backend/core-services/extension-api/tests/unit/services/email-client.service.test.ts`

- Remove: `jest.mock('imap')`, `jest.mock('mailparser')`
- Use: Real IMAP connection to test server OR mark as integration test
- Alternative: Use `imap-simple` test utilities or skip IMAP tests if no test server available
- Note: IMAP requires real email server - may need to skip or use test IMAP server

### 6. Extension Auth Middleware Tests

**File**: `backend/core-services/extension-api/tests/unit/middleware/extension-auth.middleware.test.ts`

- Remove: All database mocks
- Use: Real database connection via `getPostgreSQL()` from shared utilities
- Use: Real bcrypt for hash verification
- Add: Test API key creation in database before tests
- Add: Database cleanup after tests
- Add: Test fixtures for API keys

### 7. Rate Limit Middleware Tests

**File**: `backend/core-services/extension-api/tests/unit/middleware/rate-limit.middleware.test.ts`

- Remove: `jest.mock('cache.service')`
- Use: Real `CacheService` with test Redis
- Add: Redis cleanup between tests

## Integration Tests Refactoring

### 1. URL Check Route Tests

**File**: `backend/core-services/extension-api/tests/integration/routes/url-check.routes.test.ts`

- Remove: All service mocks
- Use: Real Express app with real services
- Use: Real database for auth middleware
- Use: Real Redis for cache and rate limiting
- Use: Real detection-api (must be running)
- Add: Test API key in database
- Add: Full request/response cycle testing

### 2. Email Scan Route Tests

**File**: `backend/core-services/extension-api/tests/integration/routes/email-scan.routes.test.ts`

- Remove: All mocks
- Use: Real services end-to-end
- Use: Real detection-api

### 3. Report Route Tests

**File**: `backend/core-services/extension-api/tests/integration/routes/report.routes.test.ts`

- Remove: `jest.mock('axios')` and service mocks
- Use: Real threat-intel service (must be running)
- Use: Real axios calls
- Add: Service availability checks

### 4. Email Client Route Tests

**File**: `backend/core-services/extension-api/tests/integration/routes/email-client.routes.test.ts`

- Remove: All mocks
- Use: Real email client service
- Note: May require test IMAP server or skip if unavailable

## Test Fixtures and Seeders

**File**: `backend/core-services/extension-api/tests/fixtures/api-keys.ts` (new)

- Test API key creation utilities
- Predefined test API keys with known hashes
- Organization and user fixtures

**File**: `backend/core-services/extension-api/tests/fixtures/database.ts` (new)

- Database seed functions
- Test data generators
- Cleanup functions

## Package.json Scripts

**File**: `backend/core-services/extension-api/package.json`

Add scripts:

- `test:docker-up`: Start test infrastructure
- `test:docker-down`: Stop test infrastructure
- `test:integration`: Run integration tests (requires services)
- `test:unit`: Run unit tests (requires test DB/Redis)
- `test:all`: Run all tests with infrastructure setup

## Docker Compose Test Configuration

**File**: `backend/core-services/extension-api/docker-compose.test.yml` (new)

```yaml
version: '3.8'
services:
  postgres-test:
    image: postgres:15-alpine
    ports:
                                                                                 - "5433:5432"
    environment:
      POSTGRES_DB: phishing_detection_test
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    
  redis-test:
    image: redis:7-alpine
    ports:
                                                                                 - "6380:6379"
    
  mongodb-test:
    image: mongo:7
    ports:
                                                                                 - "27018:27017"
```

## Environment Variables for Tests

**File**: `backend/core-services/extension-api/.env.test` (new)

```
NODE_ENV=test
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/phishing_detection_test
TEST_REDIS_URL=redis://localhost:6380
TEST_MONGODB_URL=mongodb://localhost:27018/phishing_detection_test
DETECTION_API_URL=http://localhost:3001
THREAT_INTEL_URL=http://localhost:3002
```

## Implementation Steps

1. Create Docker Compose test configuration
2. Create test helper utilities (database, Redis, services)
3. Update test setup file to initialize real connections
4. Refactor unit tests one by one (remove mocks, add real services)
5. Refactor integration tests (remove mocks, use real app)
6. Create test fixtures and seeders
7. Update package.json scripts
8. Add service availability checks
9. Run tests and fix issues
10. Ensure all tests pass

## Key Considerations

- External services (detection-api, threat-intel) must be running for integration tests
- IMAP tests may need test email server or be marked as optional
- Database migrations must run before tests
- Redis must be flushed between tests to avoid state leakage
- Test database should be separate from development database
- Tests should be idempotent and independent

## Files to Create

- `backend/core-services/extension-api/docker-compose.test.yml`
- `backend/core-services/extension-api/tests/helpers/test-db.ts`
- `backend/core-services/extension-api/tests/helpers/test-redis.ts`
- `backend/core-services/extension-api/tests/helpers/test-services.ts`
- `backend/core-services/extension-api/tests/fixtures/api-keys.ts`
- `backend/core-services/extension-api/tests/fixtures/database.ts`
- `backend/core-services/extension-api/.env.test`

## Files to Modify

- All test files in `tests/unit/` (remove mocks, add real services)
- All test files in `tests/integration/` (remove mocks, use real app)
- `tests/setup.ts` (initialize real connections)
- `package.json` (add test scripts)
- `jest.config.js` (update if needed for test environment)