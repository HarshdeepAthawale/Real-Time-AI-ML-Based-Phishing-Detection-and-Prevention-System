import request from 'supertest';
import express from 'express';
import { DataSource } from 'typeorm';
import { FeedManagerService } from '../../../src/services/feed-manager.service';
import feedsRoutes from '../../../src/routes/feeds.routes';
import { Feed } from '../../../src/services/feed-manager.service';
import { errorHandler } from '../../../src/middleware/error-handler.middleware';
import { setupTestDatabase, cleanupTestDatabase, disconnectTestDatabase, getTestDataSource, checkDatabaseAvailability, resetDatabaseAvailabilityCache, beginTestTransaction, rollbackTestTransaction } from '../../helpers/test-database';

// Check if infrastructure is available
let infrastructureAvailable = false;

describe('Feed Routes Integration Tests', () => {
  let app: express.Application;
  let testDataSource: DataSource;
  let feedManager: FeedManagerService;
  let createdFeedId: string | undefined;

  beforeAll(async () => {
    // Reset cache to ensure fresh check
    resetDatabaseAvailabilityCache();
    
    // Check if infrastructure is available
    const dbAvailable = await checkDatabaseAvailability();
    infrastructureAvailable = dbAvailable;

    if (!infrastructureAvailable) {
      console.warn(
        '\n⚠️  Skipping integration tests - database not available.\n' +
        `DB: ${dbAvailable}\n` +
        'To run integration tests:\n' +
        '1. Start PostgreSQL: postgres -D /path/to/data\n' +
        '2. Create test database: createdb phishing_detection_test\n' +
        '3. Run migrations: cd backend/shared/database && npm run migrate\n'
      );
      return;
    }

    // Setup test database
    try {
      testDataSource = await setupTestDatabase();
    } catch (error) {
      console.error('Failed to setup test database:', error);
      infrastructureAvailable = false;
    }
  });

  beforeEach(async () => {
    if (!infrastructureAvailable || !testDataSource) {
      return;
    }

    // Start transaction for this test (replaces cleanupTestDatabase)
    const queryRunner = await beginTestTransaction();
    
    // Initialize real services with query runner for transaction isolation
    feedManager = new FeedManagerService(testDataSource, queryRunner);
    
    // Setup Express app
    app = express();
    app.use(express.json());
    
    // Set real services in app
    app.set('feedManager', feedManager);
    app.use('/api/v1/feeds', feedsRoutes);
    
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
    // Disconnect
    await disconnectTestDatabase();
  });

  describe('GET /api/v1/feeds', () => {
    it('should return all feeds', async () => {
      // Create real feeds in database
      await feedManager.createFeed({
        name: 'MISP Feed',
        feedType: 'misp',
        isActive: true,
      });

      await feedManager.createFeed({
        name: 'OTX Feed',
        feedType: 'otx',
        isActive: true,
      });

      const response = await request(app)
        .get('/api/v1/feeds');

      expect(response.status).toBe(200);
      expect(response.body.feeds).toBeDefined();
      expect(Array.isArray(response.body.feeds)).toBe(true);
      expect(response.body.feeds.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('GET /api/v1/feeds/:id', () => {
    it('should return feed by ID', async () => {
      // Create real feed in database
      const createdFeed = await feedManager.createFeed({
        name: 'MISP Feed',
        feedType: 'misp',
        isActive: true,
      });

      const response = await request(app)
        .get(`/api/v1/feeds/${createdFeed.id}`);

      expect(response.status).toBe(200);
      expect(response.body).toBeDefined();
      expect(response.body.id).toBe(createdFeed.id);
      expect(response.body.name).toBe('MISP Feed');
    });

    it('should return 404 when feed not found', async () => {
      const feedId = '123e4567-e89b-12d3-a456-426614174999'; // Valid UUID that doesn't exist
      
      const response = await request(app)
        .get(`/api/v1/feeds/${feedId}`);

      expect(response.status).toBe(404);
    });
  });

  describe('POST /api/v1/feeds', () => {
    it('should create new feed', async () => {
      const response = await request(app)
        .post('/api/v1/feeds')
        .send({
          name: 'New Feed',
          feedType: 'misp',
          apiEndpoint: 'https://misp.example.com',
          apiKeyEncrypted: 'encrypted_key',
          syncIntervalMinutes: 60,
        });

      expect(response.status).toBe(201);
      expect(response.body).toBeDefined();
      expect(response.body.name).toBe('New Feed');
      expect(response.body.feedType).toBe('misp');
      expect(response.body.id).toBeDefined();
      
      // Verify it was actually created in database
      const found = await feedManager.getFeedById(response.body.id);
      expect(found).toBeDefined();
      expect(found?.name).toBe('New Feed');
      
      createdFeedId = response.body.id;
    });

    it('should validate feed creation request', async () => {
      const response = await request(app)
        .post('/api/v1/feeds')
        .send({
          name: '',
          feedType: 'invalid',
        });

      expect(response.status).toBe(400);
    });
  });

  describe('PUT /api/v1/feeds/:id', () => {
    it('should update feed', async () => {
      // Create real feed first
      const feed = await feedManager.createFeed({
        name: 'Test Feed',
        feedType: 'misp',
        isActive: true,
      });

      const response = await request(app)
        .put(`/api/v1/feeds/${feed.id}`)
        .send({
          name: 'Updated Feed',
          isActive: false,
        });

      expect(response.status).toBe(200);
      expect(response.body).toBeDefined();
      expect(response.body.name).toBe('Updated Feed');
      expect(response.body.isActive).toBe(false);
      
      // Verify it was actually updated in database
      const found = await feedManager.getFeedById(feed.id!);
      expect(found?.name).toBe('Updated Feed');
      expect(found?.isActive).toBe(false);
    });

    it('should return 404 when updating non-existent feed', async () => {
      const feedId = '123e4567-e89b-12d3-a456-426614174999'; // Valid UUID that doesn't exist
      
      const response = await request(app)
        .put(`/api/v1/feeds/${feedId}`)
        .send({ name: 'Updated' });

      expect(response.status).toBe(404);
    });
  });

  describe('DELETE /api/v1/feeds/:id', () => {
    it('should delete feed', async () => {
      // Create real feed first
      const feed = await feedManager.createFeed({
        name: 'Feed to Delete',
        feedType: 'misp',
        isActive: true,
      });

      const response = await request(app)
        .delete(`/api/v1/feeds/${feed.id}`);

      expect(response.status).toBe(204);
      
      // Verify it was actually deleted
      const found = await feedManager.getFeedById(feed.id!);
      expect(found).toBeNull();
    });
  });

  describe('POST /api/v1/feeds/:id/toggle', () => {
    it('should toggle feed active status', async () => {
      // Create real feed first
      const feed = await feedManager.createFeed({
        name: 'Test Feed',
        feedType: 'misp',
        isActive: true,
      });

      const response = await request(app)
        .post(`/api/v1/feeds/${feed.id}/toggle`);

      expect(response.status).toBe(200);
      expect(response.body).toBeDefined();
      expect(response.body.isActive).toBe(false); // Should be toggled to false
      
      // Verify it was actually toggled in database
      const found = await feedManager.getFeedById(feed.id!);
      expect(found?.isActive).toBe(false);
    });
  });
});
