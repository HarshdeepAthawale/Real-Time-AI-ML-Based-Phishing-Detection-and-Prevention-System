import request from 'supertest';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { URLCheckerService } from '../../../src/services/url-checker.service';
import { CacheService } from '../../../src/services/cache.service';
import { PrivacyFilterService } from '../../../src/services/privacy-filter.service';
import { extensionAuthMiddleware } from '../../../src/middleware/extension-auth.middleware';
import { createRateLimitMiddleware } from '../../../src/middleware/rate-limit.middleware';
import urlCheckRoutes from '../../../src/routes/url-check.routes';
import { connectTestDatabase, truncateTables } from '../../helpers/test-db';
import { flushTestRedis, getTestRedisUrl } from '../../helpers/test-redis';
import { createStandardTestFixtures, TEST_API_KEY } from '../../fixtures/api-keys';
import { isDetectionApiAvailable } from '../../helpers/test-services';

describe('URL Check Route Integration', () => {
  let app: express.Application;
  let urlChecker: URLCheckerService;
  let cacheService: CacheService;
  let privacyFilter: PrivacyFilterService;
  let dataSource: any;
  let standardFixtures: any;

  beforeAll(async () => {
    // Set up test environment
    const redisUrl = getTestRedisUrl();
    process.env.REDIS_URL = redisUrl;
    process.env.REDIS_HOST = redisUrl.split('://')[1]?.split(':')[0] || 'localhost';
    process.env.REDIS_PORT = redisUrl.split(':')[2]?.split('/')[0] || '6380';
    process.env.DATABASE_URL = process.env.TEST_DATABASE_URL || 'postgresql://postgres:postgres@localhost:5433/phishing_detection_test';
    
    // Connect to test database
    dataSource = await connectTestDatabase();
  });

  beforeEach(async () => {
    // Clean up database before each test to avoid deadlocks
    await truncateTables();
    standardFixtures = await createStandardTestFixtures(dataSource);
    
    // Flush Redis
    await flushTestRedis();
    
    // Create real service instances
    cacheService = new CacheService();
    privacyFilter = new PrivacyFilterService();
    
    // Wait for cache service to connect
    let retries = 10;
    while (retries > 0 && !(await cacheService.isConnected())) {
      await new Promise(resolve => setTimeout(resolve, 100));
      retries--;
    }
    
    urlChecker = new URLCheckerService(
      process.env.DETECTION_API_URL || 'http://localhost:3001',
      cacheService,
      privacyFilter
    );

    // Create Express app with real services
    app = express();
    app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
    app.use(cors({ origin: true, credentials: true }));
    app.use(express.json({ limit: '1mb' }));
    app.use(express.urlencoded({ extended: true }));

    // Store services in app context
    app.set('urlChecker', urlChecker);
    app.set('cacheService', cacheService);
    app.set('privacyFilter', privacyFilter);

    // Use real middleware
    app.use(
      '/api/v1/extension/check-url',
      extensionAuthMiddleware,
      createRateLimitMiddleware(cacheService),
      urlCheckRoutes
    );
  });

  afterEach(async () => {
    try {
      if (cacheService) {
        await cacheService.disconnect();
      }
    } catch (error) {
      // Ignore disconnect errors
    }
    await flushTestRedis();
  });

  describe('POST /api/v1/extension/check-url', () => {
    it('should check URL successfully', async () => {
      const response = await request(app)
        .post('/api/v1/extension/check-url')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({ url: 'https://example.com' });

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('isThreat');
      expect(response.body).toHaveProperty('cached');
      expect(response.body).toHaveProperty('timestamp');
    });

    it('should validate URL format', async () => {
      const response = await request(app)
        .post('/api/v1/extension/check-url')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({ url: 'not-a-valid-url' });

      expect(response.status).toBe(400);
      expect(response.body.error.message).toBe('Validation failed');
    });

    it('should handle optional parameters', async () => {
      const response = await request(app)
        .post('/api/v1/extension/check-url')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({
          url: 'https://example.com',
          includeFullAnalysis: true,
          privacyMode: true,
          pageText: 'Page content',
          screenshot: 'base64image',
        });

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('isThreat');
    });

    it('should handle service unavailable', async () => {
      app.set('urlChecker', null);

      const response = await request(app)
        .post('/api/v1/extension/check-url')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({ url: 'https://example.com' });

      expect(response.status).toBe(500);
      expect(response.body.error.message).toBe('Service unavailable');
    });

    it('should require URL parameter', async () => {
      const response = await request(app)
        .post('/api/v1/extension/check-url')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({});

      expect(response.status).toBe(400);
    });

    it('should require API key', async () => {
      const response = await request(app)
        .post('/api/v1/extension/check-url')
        .send({ url: 'https://example.com' });

      expect(response.status).toBe(401);
      expect(response.body.error.message).toBe('API key required');
    });
  });
});
