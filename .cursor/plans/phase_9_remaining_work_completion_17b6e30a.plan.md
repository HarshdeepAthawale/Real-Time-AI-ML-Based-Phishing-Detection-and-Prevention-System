---
name: Phase 9 Remaining Work Completion
overview: "Complete the remaining Phase 9 deliverables: comprehensive test suite, API key database validation, and extension icons. All components will be production-ready with proper error handling."
todos:
  - id: "1"
    content: Add TypeORM, bcrypt, and testing dependencies to package.json
    status: completed
  - id: "2"
    content: Set up database connection in extension-api index.ts using shared utilities
    status: completed
  - id: "3"
    content: Implement API key database validation in extension-auth.middleware.ts
    status: completed
  - id: "4"
    content: Create jest.config.js with TypeScript and coverage configuration
    status: completed
  - id: "5"
    content: Write unit tests for URL checker service
    status: completed
  - id: "6"
    content: Write unit tests for email scanner service
    status: completed
  - id: "7"
    content: Write unit tests for privacy filter service
    status: completed
  - id: "8"
    content: Write unit tests for cache service
    status: completed
  - id: "9"
    content: Write unit tests for email client service
    status: completed
  - id: "10"
    content: Write unit tests for extension auth middleware
    status: completed
  - id: "11"
    content: Write unit tests for rate limit middleware
    status: completed
  - id: "12"
    content: Write integration tests for URL check route
    status: completed
  - id: "13"
    content: Write integration tests for email scan route
    status: completed
  - id: "14"
    content: Write integration tests for report route
    status: completed
  - id: "15"
    content: Write integration tests for email client routes
    status: completed
  - id: "16"
    content: Create extension icons (16x16, 48x48, 128x128) for Chrome, Firefox, Edge
    status: completed
  - id: "17"
    content: Run test suite and verify all tests pass
    status: completed
  - id: "18"
    content: Generate test coverage report and verify minimum 80% coverage
    status: completed
---

# Phase 9 Remaining Work Completion Plan

## Overview

Complete the remaining 10-15% of Phase 9 implementation: comprehensive test suite, API key database validation, and extension icons.

## Remaining Tasks

### 1. API Key Database Validation

**File**: `backend/core-services/extension-api/src/middleware/extension-auth.middleware.ts`

**Current State**: Accepts any API key format, has TODO comment

**Implementation**:

- Add TypeORM dependency to package.json
- Import shared database connection utilities
- Query ApiKey table using key_prefix for initial lookup
- Use bcrypt.compare to verify key_hash
- Check expiration (expires_at)
- Check revocation (revoked_at is null)
- Update last_used_at timestamp
- Handle database connection errors gracefully (fail fast if DB unavailable per requirement)

**Dependencies to Add**:

- `typeorm` - ORM for database access
- `bcrypt` - Password/key hashing verification
- `@types/bcrypt` - TypeScript types

**Database Connection**:

- Use `backend/shared/database/connection.ts` utilities
- Initialize connection in `index.ts` startup
- Handle connection failures during startup

### 2. Comprehensive Test Suite

**Test Structure** (following detection-api pattern):

```
backend/core-services/extension-api/tests/
├── unit/
│   ├── services/
│   │   ├── url-checker.service.test.ts
│   │   ├── email-scanner.service.test.ts
│   │   ├── privacy-filter.service.test.ts
│   │   ├── cache.service.test.ts
│   │   └── email-client.service.test.ts
│   ├── middleware/
│   │   ├── extension-auth.middleware.test.ts
│   │   └── rate-limit.middleware.test.ts
│   └── utils/
│       └── (if any utility functions)
└── integration/
    ├── routes/
    │   ├── url-check.routes.test.ts
    │   ├── email-scan.routes.test.ts
    │   ├── report.routes.test.ts
    │   └── email-client.routes.test.ts
    └── services/
        └── (integration tests for service interactions)
```

**Test Coverage Requirements**:

**Unit Tests**:

- URL Checker Service: cache hits/misses, privacy mode, error handling, detection API integration
- Email Scanner Service: email parsing, link extraction, privacy filtering, threat detection
- Privacy Filter Service: URL filtering, email content filtering, edge cases
- Cache Service: get/set operations, TTL handling, Redis connection failures
- Email Client Service: IMAP connection, email parsing, threat detection events
- Auth Middleware: API key validation, missing key handling, database validation
- Rate Limit Middleware: request counting, TTL handling, Redis failures

**Integration Tests**:

- URL check route: full request/response flow, error handling
- Email scan route: full request/response flow, link checking
- Report route: threat intelligence integration, error handling
- Email client routes: connection management, scanning triggers

**Test Setup**:

- Jest configuration (already in package.json)
- Mock Redis for cache tests
- Mock axios for HTTP calls
- Mock IMAP for email client tests
- Mock TypeORM repositories for database tests

**Additional Dependencies**:

- `@types/jest` - Already present
- `ts-jest` - TypeScript support for Jest
- `jest-mock-extended` - Enhanced mocking capabilities

### 3. Extension Icons

**Files to Create**:

- `extensions/chrome/icons/icon16.png`
- `extensions/chrome/icons/icon48.png`
- `extensions/chrome/icons/icon128.png`
- `extensions/firefox/icons/icon16.png`
- `extensions/firefox/icons/icon48.png`
- `extensions/firefox/icons/icon128.png`
- `extensions/edge/icons/icon16.png`
- `extensions/edge/icons/icon48.png`
- `extensions/edge/icons/icon128.png`

**Implementation Approach**:

- Create simple SVG-based security shield icon
- Convert to PNG at required sizes (16x16, 48x48, 128x128)
- Use shield/checkmark design indicating security/protection
- Color scheme: Green shield with checkmark (safe) or red shield with warning (threat)

**Tools Needed**:

- SVG creation (programmatic or manual)
- Image conversion tool (ImageMagick, sharp, or similar)
- Or use a simple script to generate icons

**Alternative**: Create a simple Node.js script using `sharp` or `canvas` to generate icons programmatically

### 4. Jest Configuration

**File**: `backend/core-services/extension-api/jest.config.js`

**Configuration**:

- TypeScript support via ts-jest
- Test environment setup
- Coverage reporting
- Mock path configuration
- Test timeout settings

### 5. Database Connection Setup

**File**: `backend/core-services/extension-api/src/index.ts`

**Changes**:

- Import `connectPostgreSQL` from shared database utilities
- Initialize database connection on startup
- Handle connection errors
- Add database health check to `/health` endpoint
- Graceful shutdown: disconnect database on SIGTERM/SIGINT

**Dependencies**:

- Ensure TypeORM entities are accessible
- Import ApiKey model from shared models

## File Structure

```
backend/core-services/extension-api/
├── src/
│   ├── index.ts (add DB connection)
│   ├── middleware/
│   │   └── extension-auth.middleware.ts (add DB validation)
│   └── services/
│       └── (existing services)
├── tests/
│   ├── unit/
│   │   ├── services/
│   │   ├── middleware/
│   │   └── utils/
│   └── integration/
│       └── routes/
├── jest.config.js (new)
└── package.json (add dependencies)

extensions/
├── chrome/icons/ (replace placeholders)
├── firefox/icons/ (replace placeholders)
└── edge/icons/ (replace placeholders)
```

## Dependencies to Add

**Backend**:

- `typeorm` - Database ORM
- `bcrypt` - Password/key hashing
- `@types/bcrypt` - TypeScript types
- `ts-jest` - Jest TypeScript support
- `jest-mock-extended` - Enhanced mocking
- `sharp` or `canvas` - Icon generation (optional, if programmatic)

## Implementation Steps

1. **Add Dependencies**: Update package.json with required packages
2. **Database Connection**: Add TypeORM setup in index.ts
3. **API Key Validation**: Implement database validation in auth middleware
4. **Jest Configuration**: Create jest.config.js
5. **Unit Tests**: Write tests for all services and middleware
6. **Integration Tests**: Write tests for all routes
7. **Extension Icons**: Generate or create icon files
8. **Test Execution**: Verify all tests pass
9. **Documentation**: Update README with test instructions

## Testing Strategy

- **Unit Tests**: Mock all external dependencies (Redis, HTTP clients, IMAP, Database)
- **Integration Tests**: Use test database and mock external services
- **Coverage Target**: Minimum 80% code coverage
- **Test Data**: Use fixtures for consistent testing

## Key Considerations

1. **Database Validation**: Fail fast if database unavailable (per requirement)
2. **Icon Design**: Simple, recognizable security shield icon
3. **Test Isolation**: Each test should be independent
4. **Mock Strategy**: Mock external services, test internal logic
5. **Error Handling**: Test error paths and edge cases

## Deliverables Checklist

- [ ] API key database validation implemented
- [ ] Database connection setup in extension-api
- [ ] Jest configuration file created
- [ ] Unit tests for all services (5 services)
- [ ] Unit tests for all middleware (2 middleware)
- [ ] Integration tests for all routes (4 routes)
- [ ] Extension icons created (9 icon files)
- [ ] All tests passing
- [ ] Test coverage report generated
- [ ] Documentation updated