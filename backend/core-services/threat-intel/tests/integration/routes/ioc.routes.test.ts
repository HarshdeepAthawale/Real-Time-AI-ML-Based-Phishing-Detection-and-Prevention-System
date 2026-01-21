import request from 'supertest';
import express from 'express';
import { DataSource } from 'typeorm';
import Redis from 'ioredis';
import { IOCMatcherService } from '../../../src/services/ioc-matcher.service';
import { IOCManagerService } from '../../../src/services/ioc-manager.service';
import { EnrichmentService } from '../../../src/services/enrichment.service';
import iocRoutes from '../../../src/routes/ioc.routes';
import { IOC } from '../../../src/models/ioc.model';
import { errorHandler } from '../../../src/middleware/error-handler.middleware';
import { setupTestDatabase, cleanupTestDatabase, disconnectTestDatabase, getTestDataSource, checkDatabaseAvailability, resetDatabaseAvailabilityCache, beginTestTransaction, rollbackTestTransaction, getTestQueryRunner } from '../../helpers/test-database';
import { setupTestRedis, cleanupTestRedis, disconnectTestRedis, getTestRedis, checkRedisAvailability, resetRedisAvailabilityCache } from '../../helpers/test-redis';

describe('IOC Routes Integration Tests', () => {
  let app: express.Application;
  let testDataSource: DataSource;
  let testRedis: Redis;
  let iocManager: IOCManagerService;
  let iocMatcher: IOCMatcherService;
  let enrichmentService: EnrichmentService;

  beforeAll(async () => {
    // Reset caches to ensure fresh check
    resetDatabaseAvailabilityCache();
    resetRedisAvailabilityCache();
    
    // Check if infrastructure is available
    const dbAvailable = await checkDatabaseAvailability();
    const redisAvailable = await checkRedisAvailability();
    
    if (!dbAvailable || !redisAvailable) {
      console.warn(
        '\n⚠️  Skipping integration tests - database or Redis not available.\n' +
        `DB: ${dbAvailable}, Redis: ${redisAvailable}\n` +
        'TEST_DATABASE_URL:', process.env.TEST_DATABASE_URL || 'not set', '\n' +
        'TEST_REDIS_URL:', process.env.TEST_REDIS_URL || 'not set', '\n' +
        'To run integration tests:\n' +
        '1. Start PostgreSQL: postgres -D /path/to/data\n' +
        '2. Start Redis: redis-server\n' +
        '3. Create test database: createdb phishing_detection_test\n' +
        '4. Run migrations: cd backend/shared/database && npm run migrate\n'
      );
      // Skip all tests in this suite
      return;
    }

    // Setup test database and Redis
    testDataSource = await setupTestDatabase();
    testRedis = await setupTestRedis();
  });

  beforeEach(async () => {
    if (!testDataSource || !testRedis) {
      return;
    }

    // Clean Redis (Redis doesn't support transactions)
    await cleanupTestRedis();
    
    // Start transaction for this test (replaces cleanupTestDatabase)
    const queryRunner = await beginTestTransaction();
    
    // Initialize real services with query runner for transaction isolation
    iocManager = new IOCManagerService(testDataSource, queryRunner);
    iocMatcher = new IOCMatcherService(testRedis, iocManager);
    enrichmentService = new EnrichmentService(iocManager);
    
    // Initialize bloom filters
    await iocMatcher.initializeBloomFilters();
    
    // Setup Express app
    app = express();
    app.use(express.json());
    
    // Set real services in app
    app.set('iocManager', iocManager);
    app.set('iocMatcher', iocMatcher);
    app.set('enrichmentService', enrichmentService);

    app.use('/api/v1/ioc', iocRoutes);
    
    // Error handler MUST be last middleware
    app.use(errorHandler);
  });

  afterEach(async () => {
    // Rollback transaction (cleanup happens automatically)
    await rollbackTestTransaction();
  });

  afterAll(async () => {
    // Rollback any active transaction
    await rollbackTestTransaction();
    // Cleanup and disconnect
    await cleanupTestRedis();
    await disconnectTestDatabase();
    await disconnectTestRedis();
  });

  describe('POST /api/v1/ioc/check', () => {
    it('should return found: true when IOC exists', async () => {
      // Create real IOC in database
      const testIOC = await iocManager.createIOC({
        iocType: 'domain',
        iocValue: 'malicious.example.com',
        threatType: 'phishing',
        severity: 'high',
        confidence: 90,
        source: 'feed',
      });

      // Verify IOC was actually saved
      const verifyIOC = await iocManager.findIOC('domain', 'malicious.example.com');
      if (!verifyIOC) {
        throw new Error('IOC was not saved to database - test setup issue');
      }

      // Add to bloom filter
      await iocMatcher.addToBloomFilter('domain', 'malicious.example.com');

      const response = await request(app)
        .post('/api/v1/ioc/check')
        .send({
          iocType: 'domain',
          iocValue: 'malicious.example.com',
        });

      expect(response.status).toBe(200);
      expect(response.body.found).toBe(true);
      expect(response.body.ioc).toBeDefined();
      expect(response.body.ioc.iocValue).toBe('malicious.example.com');
    });

    it('should return found: false when IOC does not exist', async () => {
      const response = await request(app)
        .post('/api/v1/ioc/check')
        .send({
          iocType: 'domain',
          iocValue: 'safe.example.com',
        });

      expect(response.status).toBe(200);
      expect(response.body.found).toBe(false);
      expect(response.body.ioc).toBeUndefined();
    });

    it('should validate request body', async () => {
      const response = await request(app)
        .post('/api/v1/ioc/check')
        .send({
          iocType: 'invalid_type',
          iocValue: 'test',
        });

      expect(response.status).toBe(400);
    });

    it('should enrich IOC when enrich=true query param is provided', async () => {
      // Create IOC in database
      const testIOC = await iocManager.createIOC({
        iocType: 'domain',
        iocValue: 'malicious.example.com',
        threatType: 'phishing',
        severity: 'high',
        confidence: 90,
        source: 'feed',
      });

      // Verify IOC was actually saved
      const verifyIOC = await iocManager.findIOC('domain', 'malicious.example.com');
      if (!verifyIOC) {
        throw new Error('IOC was not saved to database - test setup issue');
      }

      await iocMatcher.addToBloomFilter('domain', 'malicious.example.com');

      const response = await request(app)
        .post('/api/v1/ioc/check?enrich=true')
        .send({
          iocType: 'domain',
          iocValue: 'malicious.example.com',
        });

      expect(response.status).toBe(200);
      expect(response.body.found).toBe(true);
      // Enrichment may or may not return additional context depending on related IOCs
    });
  });

  describe('POST /api/v1/ioc/bulk-check', () => {
    it('should check multiple IOCs', async () => {
      // Create real IOC in database
      await iocManager.createIOC({
        iocType: 'domain',
        iocValue: 'malicious1.com',
        threatType: 'phishing',
        severity: 'high',
        source: 'feed',
      });
      await iocMatcher.addToBloomFilter('domain', 'malicious1.com');

      const response = await request(app)
        .post('/api/v1/ioc/bulk-check')
        .send({
          iocs: [
            { iocType: 'domain', iocValue: 'malicious1.com' },
            { iocType: 'ip', iocValue: '192.168.1.1' },
          ],
        });

      expect(response.status).toBe(200);
      expect(response.body.results).toHaveLength(2);
      expect(response.body.results[0].found).toBe(true);
      expect(response.body.results[1].found).toBe(false);
      expect(response.body.summary.total).toBe(2);
      expect(response.body.summary.found).toBe(1);
      expect(response.body.summary.notFound).toBe(1);
    });

    it('should validate bulk check request', async () => {
      const response = await request(app)
        .post('/api/v1/ioc/bulk-check')
        .send({
          iocs: [],
        });

      expect(response.status).toBe(400);
    });
  });

  describe('POST /api/v1/ioc/report', () => {
    it('should create new IOC', async () => {
      const response = await request(app)
        .post('/api/v1/ioc/report')
        .send({
          iocType: 'domain',
          iocValue: 'reported.example.com',
          threatType: 'phishing',
          severity: 'medium',
          confidence: 70,
        });

      expect(response.status).toBe(201);
      expect(response.body).toBeDefined();
      expect(response.body.iocType).toBe('domain');
      expect(response.body.iocValue).toBe('reported.example.com');
      expect(response.body.source).toBe('user');
      
      // Verify it was actually created in database
      const found = await iocManager.findIOC('domain', 'reported.example.com');
      expect(found).toBeDefined();
      expect(found?.iocValue).toBe('reported.example.com');
    });
  });

  describe('GET /api/v1/ioc/search', () => {
    it('should search IOCs with filters', async () => {
      // Create test IOCs
      await iocManager.createIOC({
        iocType: 'domain',
        iocValue: 'malicious1.com',
        threatType: 'phishing',
        severity: 'high',
        source: 'feed',
      });

      await iocManager.createIOC({
        iocType: 'ip',
        iocValue: '192.168.1.1',
        threatType: 'malware',
        severity: 'high',
        source: 'feed',
      });

      const response = await request(app)
        .get('/api/v1/ioc/search')
        .query({
          iocType: 'domain',
          severity: 'high',
          limit: 10,
          offset: 0,
        });

      expect(response.status).toBe(200);
      expect(response.body.iocs).toBeDefined();
      expect(Array.isArray(response.body.iocs)).toBe(true);
      expect(response.body.total).toBeGreaterThanOrEqual(1);
      
      // Verify all returned IOCs match the filter
      if (response.body.iocs.length > 0) {
        expect(response.body.iocs[0].iocType).toBe('domain');
      }
    });
  });

  describe('GET /api/v1/ioc/stats', () => {
    it('should return IOC statistics', async () => {
      // Create test IOCs
      await iocManager.createIOC({
        iocType: 'domain',
        iocValue: 'malicious1.com',
        threatType: 'phishing',
        severity: 'high',
        source: 'feed',
      });

      await iocManager.createIOC({
        iocType: 'ip',
        iocValue: '192.168.1.1',
        threatType: 'malware',
        severity: 'medium',
        source: 'feed',
      });

      const response = await request(app)
        .get('/api/v1/ioc/stats');

      expect(response.status).toBe(200);
      expect(response.body.total).toBeGreaterThanOrEqual(2);
      expect(response.body.byType).toBeDefined();
      expect(response.body.bySeverity).toBeDefined();
    });
  });
});
