# Threat Intelligence Service - Test Documentation

This directory contains the test suite for the Threat Intelligence Service.

## Test Structure

```
tests/
├── unit/                    # Unit tests (no external dependencies)
│   ├── integrations/        # Feed client tests
│   ├── services/            # Service tests
│   └── utils/               # Utility function tests
├── integration/             # Integration tests (require database/Redis)
│   └── routes/              # API route tests
├── helpers/                 # Test utilities and helpers
│   ├── test-database.ts    # Database setup/teardown
│   ├── test-redis.ts       # Redis setup/teardown
│   ├── test-app.ts         # Express app setup
│   └── test-setup.ts       # Global test configuration
├── fixtures/                # Test data fixtures
└── jest.config.ts           # Jest configuration
```

## Running Tests

### Prerequisites

**For Unit Tests:**
- Node.js 20+
- npm dependencies installed (`npm install`)

**For Integration Tests:**
- All unit test prerequisites
- PostgreSQL 15+ running and accessible
- Redis 7+ running and accessible
- Test database created: `phishing_detection_test`
- Database migrations run (tables must exist)

### Quick Start

```bash
# Run all tests (unit + integration)
npm test

# Run only unit tests (no database/Redis required)
npm run test:unit

# Run only integration tests (requires database/Redis)
npm run test:integration

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run all tests with verbose output and coverage
npm run test:all
```

## Setting Up Integration Tests

### Quick Setup (Automated)

Use the provided setup script:

```bash
cd backend/core-services/threat-intel
npm run test:setup
```

This script will:
- Check if PostgreSQL and Redis are running
- Create the test database (`phishing_detection_test`)
- Run database migrations
- Verify everything is ready

### Manual Setup

#### 1. Start PostgreSQL

```bash
# Using Docker
docker run -d \
  --name postgres-test \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=phishing_detection_test \
  -p 5432:5432 \
  postgres:15-alpine

# Or using local PostgreSQL
createdb phishing_detection_test
```

#### 2. Start Redis

```bash
# Using Docker
docker run -d \
  --name redis-test \
  -p 6379:6379 \
  redis:7-alpine

# Or using local Redis
redis-server
```

#### 3. Run Database Migrations

```bash
# From the backend/shared/database directory
cd ../../shared/database
npm install  # If not already installed
npm run build  # Build migrations
npm run migration:run

# Or set DATABASE_URL and run migrations
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/phishing_detection_test
npm run migration:run
```

### 4. Configure Environment Variables (Optional)

```bash
# Set custom test database URL
export TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/phishing_detection_test

# Set custom test Redis URL
export TEST_REDIS_URL=redis://localhost:6379/1

# Or use the main database/Redis URLs (will use test database based on name)
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/phishing_detection
export REDIS_URL=redis://localhost:6379
```

## Test Behavior

### Unit Tests

- **No external dependencies** - All external services are mocked
- **Fast execution** - Run in milliseconds
- **Always run** - These tests will always execute

### Integration Tests

- **Require infrastructure** - Need PostgreSQL and Redis
- **Skip automatically** - If database/Redis unavailable, tests are skipped with helpful messages
- **Real services** - Use actual database and Redis connections
- **Isolated** - Each test cleans up data before and after

## Test Infrastructure

### Database Helper (`test-database.ts`)

- `setupTestDatabase()` - Creates database connection
- `cleanupTestDatabase()` - Cleans test data (truncates tables)
- `disconnectTestDatabase()` - Closes database connection
- `checkDatabaseAvailability()` - Checks if database is available
- `getTestDataSource()` - Gets active database connection

### Redis Helper (`test-redis.ts`)

- `setupTestRedis()` - Creates Redis connection
- `cleanupTestRedis()` - Cleans test data (removes test keys)
- `disconnectTestRedis()` - Closes Redis connection
- `checkRedisAvailability()` - Checks if Redis is available
- `getTestRedis()` - Gets active Redis connection

### Test Isolation

- Each test suite uses a separate database connection
- Redis keys are prefixed with `test:threat-intel:` for isolation
- Data is cleaned before each test (`beforeEach`)
- Connections are closed after all tests (`afterAll`)

## Troubleshooting

### Integration Tests Are Skipped

**Problem:** Tests show as skipped with warning message

**Solutions:**
1. Check PostgreSQL is running: `pg_isready` or `docker ps`
2. Check Redis is running: `redis-cli ping` or `docker ps`
3. Verify database exists: `psql -l | grep phishing_detection_test`
4. Check connection strings in environment variables
5. Verify network connectivity to database/Redis

### Database Connection Errors

**Problem:** `ECONNREFUSED` or connection timeout errors

**Solutions:**
1. Verify PostgreSQL is listening on correct port (default: 5432)
2. Check firewall settings
3. Verify `DATABASE_URL` or `TEST_DATABASE_URL` is correct
4. Check PostgreSQL logs for authentication issues
5. Ensure database user has proper permissions

### Redis Connection Errors

**Problem:** Redis connection failures

**Solutions:**
1. Verify Redis is listening on correct port (default: 6379)
2. Check Redis configuration allows connections
3. Verify `REDIS_URL` or `TEST_REDIS_URL` is correct
4. Check Redis logs for errors
5. Ensure Redis is not password-protected (or update connection string)

### Test Database Not Found

**Problem:** `database "phishing_detection_test" does not exist`

**Solutions:**
1. Create test database: `createdb phishing_detection_test`
2. Or set `TEST_DATABASE_URL` to use existing database
3. Ensure PostgreSQL user has CREATE DATABASE permission

### Migration Errors

**Problem:** Tables don't exist or migration errors

**Solutions:**
1. Run migrations: `cd backend/shared/database && npm run migrate`
2. Or manually create tables from schema files
3. Verify database user has CREATE TABLE permission
4. Check migration logs for specific errors

### Test Timeout

**Problem:** Tests timeout before completing

**Solutions:**
1. Increase Jest timeout in `jest.config.ts` (default: 30000ms)
2. Check database/Redis performance
3. Verify network latency isn't too high
4. Check for long-running queries blocking tests

## Writing Tests

### Unit Test Example

```typescript
import { normalizeURL } from '../../../src/utils/normalizers';

describe('URL Normalizer', () => {
  it('should normalize URL correctly', () => {
    expect(normalizeURL('https://example.com/path')).toBe('example.com/path');
  });
});
```

### Integration Test Example

```typescript
import { setupTestDatabase, cleanupTestDatabase } from '../../helpers/test-database';
import { setupTestRedis, cleanupTestRedis } from '../../helpers/test-redis';

describe('IOC Routes', () => {
  beforeAll(async () => {
    await setupTestDatabase();
    await setupTestRedis();
  });

  beforeEach(async () => {
    await cleanupTestDatabase();
    await cleanupTestRedis();
  });

  it('should create IOC', async () => {
    // Test implementation
  });
});
```

## Best Practices

1. **Use descriptive test names** - Test names should clearly describe what is being tested
2. **Keep tests isolated** - Each test should be independent and not rely on other tests
3. **Clean up data** - Always clean test data in `beforeEach` and `afterAll`
4. **Mock external APIs** - Mock MISP/OTX clients in integration tests
5. **Test error cases** - Include tests for error scenarios and edge cases
6. **Use fixtures** - Use test fixtures for consistent test data
7. **Avoid test interdependencies** - Don't rely on test execution order

## Coverage Goals

- **Unit Tests:** > 80% coverage
- **Integration Tests:** Cover all API endpoints and critical paths
- **Critical Services:** 100% coverage for IOC Manager, IOC Matcher, Sync Service

## Continuous Integration

Tests are configured to run in CI/CD pipelines:

- Unit tests run automatically (no setup required)
- Integration tests require PostgreSQL and Redis services
- Tests are skipped gracefully if infrastructure unavailable
- CI should provision test database and Redis for full test coverage

## Additional Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [TypeORM Testing](https://typeorm.io/testing)
- [Supertest Documentation](https://github.com/visionmedia/supertest)
