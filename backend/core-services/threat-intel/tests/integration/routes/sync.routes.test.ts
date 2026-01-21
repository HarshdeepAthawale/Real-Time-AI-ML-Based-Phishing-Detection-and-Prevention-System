import request from 'supertest';
import express from 'express';
import { DataSource } from 'typeorm';
import Redis from 'ioredis';
import { SyncService } from '../../../src/services/sync.service';
import { IOCManagerService } from '../../../src/services/ioc-manager.service';
import { IOCMatcherService } from '../../../src/services/ioc-matcher.service';
import { FeedManagerService } from '../../../src/services/feed-manager.service';
import { MISPClient } from '../../../src/integrations/misp.client';
import { OTXClient } from '../../../src/integrations/otx.client';
import syncRoutes from '../../../src/routes/sync.routes';
import { errorHandler } from '../../../src/middleware/error-handler.middleware';
import { NotFoundError } from '../../../src/middleware/error-handler.middleware';
import { setupTestDatabase, cleanupTestDatabase, disconnectTestDatabase, getTestDataSource, checkDatabaseAvailability, resetDatabaseAvailabilityCache, beginTestTransaction, rollbackTestTransaction } from '../../helpers/test-database';
import { setupTestRedis, cleanupTestRedis, disconnectTestRedis, getTestRedis, checkRedisAvailability, resetRedisAvailabilityCache } from '../../helpers/test-redis';

// Mock external API clients (MISP, OTX)
jest.mock('../../../src/integrations/misp.client');
jest.mock('../../../src/integrations/otx.client');

// Check if infrastructure is available
let infrastructureAvailable = false;

describe('Sync Routes Integration Tests', () => {
  let app: express.Application;
  let testDataSource: DataSource;
  let testRedis: Redis;
  let iocManager: IOCManagerService;
  let iocMatcher: IOCMatcherService;
  let feedManager: FeedManagerService;
  let syncService: SyncService;

  beforeAll(async () => {
    // Reset caches to ensure fresh check
    resetDatabaseAvailabilityCache();
    resetRedisAvailabilityCache();
    
    // Check if infrastructure is available
    const dbAvailable = await checkDatabaseAvailability();
    const redisAvailable = await checkRedisAvailability();
    infrastructureAvailable = dbAvailable && redisAvailable;

    if (!infrastructureAvailable) {
      console.warn(
        '\n⚠️  Skipping integration tests - database or Redis not available.\n' +
        `DB: ${dbAvailable}, Redis: ${redisAvailable}\n` +
        'To run integration tests:\n' +
        '1. Start PostgreSQL: postgres -D /path/to/data\n' +
        '2. Start Redis: redis-server\n' +
        '3. Create test database: createdb phishing_detection_test\n' +
        '4. Run migrations: cd backend/shared/database && npm run migrate\n'
      );
      return;
    }

    // Setup test database and Redis
    try {
      testDataSource = await setupTestDatabase();
      testRedis = await setupTestRedis();
    } catch (error) {
      console.error('Failed to setup test infrastructure:', error);
      infrastructureAvailable = false;
    }
  });

  beforeEach(async () => {
    if (!infrastructureAvailable || !testDataSource || !testRedis) {
      return;
    }

    // Clean Redis (Redis doesn't support transactions)
    await cleanupTestRedis();
    
    // Start transaction for this test (replaces cleanupTestDatabase)
    const queryRunner = await beginTestTransaction();
    
    // Initialize real services with query runner for transaction isolation
    iocManager = new IOCManagerService(testDataSource, queryRunner);
    iocMatcher = new IOCMatcherService(testRedis, iocManager);
    feedManager = new FeedManagerService(testDataSource, queryRunner);
    
    // Initialize bloom filters
    await iocMatcher.initializeBloomFilters();
    
    // Create SyncService with mocked external clients
    syncService = new SyncService(iocManager, iocMatcher, feedManager);
    
    // Setup Express app
    app = express();
    app.use(express.json());
    
    // Set real services in app
    app.set('syncService', syncService);
    app.use('/api/v1/sync', syncRoutes);
    
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

  describe('POST /api/v1/sync/all', () => {
    it('should sync all feeds', async () => {
      // Create real feeds in database
      const feed1 = await feedManager.createFeed({
        name: 'MISP Feed',
        feedType: 'misp',
        apiEndpoint: 'https://misp.example.com',
        apiKeyEncrypted: 'test-key',
        isActive: true,
      });

      const feed2 = await feedManager.createFeed({
        name: 'OTX Feed',
        feedType: 'otx',
        apiKeyEncrypted: 'test-key',
        isActive: true,
      });

      // Mock the feed clients (external APIs)
      const mockMISPClient = {
        fetchIOCs: jest.fn().mockResolvedValue([]),
      };

      const mockOTXClient = {
        fetchIOCs: jest.fn().mockResolvedValue([]),
      };

      // Override the feed client creation in syncService
      (syncService as any).feedClients.set(feed1.id!, mockMISPClient as any);
      (syncService as any).feedClients.set(feed2.id!, mockOTXClient as any);

      const response = await request(app)
        .post('/api/v1/sync/all');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.results).toBeDefined();
      expect(Array.isArray(response.body.results)).toBe(true);
    });

    it('should handle partial failures', async () => {
      // Create real feeds
      const feed1 = await feedManager.createFeed({
        name: 'MISP Feed',
        feedType: 'misp',
        apiEndpoint: 'https://misp.example.com',
        apiKeyEncrypted: 'test-key',
        isActive: true,
      });

      const feed2 = await feedManager.createFeed({
        name: 'OTX Feed',
        feedType: 'otx',
        apiKeyEncrypted: 'test-key',
        isActive: true,
      });

      // Mock one success, one failure
      const mockMISPClient = {
        fetchIOCs: jest.fn().mockResolvedValue([]),
      };

      const mockOTXClient = {
        fetchIOCs: jest.fn().mockRejectedValue(new Error('Connection timeout')),
      };

      (syncService as any).feedClients.set(feed1.id!, mockMISPClient as any);
      (syncService as any).feedClients.set(feed2.id!, mockOTXClient as any);

      const response = await request(app)
        .post('/api/v1/sync/all');

      expect(response.status).toBe(200);
      expect(response.body.results).toBeDefined();
      // At least one result should exist
      expect(response.body.results.length).toBeGreaterThan(0);
    });
  });

  describe('POST /api/v1/sync/:feedId', () => {
    it('should sync specific feed', async () => {
      // Create real feed
      const feed = await feedManager.createFeed({
        name: 'MISP Feed',
        feedType: 'misp',
        apiEndpoint: 'https://misp.example.com',
        apiKeyEncrypted: 'test-key',
        isActive: true,
      });

      // Mock external API
      const mockMISPClient = {
        fetchIOCs: jest.fn().mockResolvedValue([
          {
            iocType: 'domain',
            iocValue: 'test-malicious.com',
            threatType: 'phishing',
            severity: 'high',
            source: 'misp',
          },
        ]),
      };

      (syncService as any).feedClients.set(feed.id!, mockMISPClient as any);

      const response = await request(app)
        .post(`/api/v1/sync/${feed.id}`);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.result).toBeDefined();
      expect(response.body.result.feedId).toBe(feed.id);
    });

    it('should return 404 when feed not found', async () => {
      const feedId = '123e4567-e89b-12d3-a456-426614174999'; // Valid UUID that doesn't exist
      
      const response = await request(app)
        .post(`/api/v1/sync/${feedId}`);

      expect(response.status).toBe(404);
    });
  });

  describe('GET /api/v1/sync/status', () => {
    it('should return sync status', async () => {
      // Create real feeds
      await feedManager.createFeed({
        name: 'MISP Feed',
        feedType: 'misp',
        isActive: true,
      });

      const response = await request(app)
        .get('/api/v1/sync/status');

      expect(response.status).toBe(200);
      expect(response.body.totalFeeds).toBeGreaterThanOrEqual(1);
      expect(response.body.activeFeeds).toBeGreaterThanOrEqual(1);
      expect(response.body.feeds).toBeDefined();
      expect(Array.isArray(response.body.feeds)).toBe(true);
    });
  });
});
