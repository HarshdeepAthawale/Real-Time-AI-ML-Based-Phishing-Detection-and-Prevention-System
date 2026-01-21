import request from 'supertest';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { extensionAuthMiddleware } from '../../../src/middleware/extension-auth.middleware';
import { createRateLimitMiddleware } from '../../../src/middleware/rate-limit.middleware';
import { CacheService } from '../../../src/services/cache.service';
import reportRoutes from '../../../src/routes/report.routes';
import { connectTestDatabase, truncateTables } from '../../helpers/test-db';
import { flushTestRedis, getTestRedisUrl } from '../../helpers/test-redis';
import { createStandardTestFixtures, TEST_API_KEY } from '../../fixtures/api-keys';
import { isThreatIntelAvailable } from '../../helpers/test-services';

describe('Report Route Integration', () => {
  let app: express.Application;
  let cacheService: CacheService;
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
    
    // Create real cache service
    cacheService = new CacheService();
    
    // Wait for cache service to connect
    let retries = 10;
    while (retries > 0 && !(await cacheService.isConnected())) {
      await new Promise(resolve => setTimeout(resolve, 100));
      retries--;
    }

    // Create Express app
    app = express();
    app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
    app.use(cors({ origin: true, credentials: true }));
    app.use(express.json({ limit: '1mb' }));
    app.use(express.urlencoded({ extended: true }));

    app.set('cacheService', cacheService);

    app.use(
      '/api/v1/extension/report',
      extensionAuthMiddleware,
      createRateLimitMiddleware(cacheService),
      reportRoutes
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

  describe('POST /api/v1/extension/report', () => {
    it('should report URL threat successfully', async () => {
      // Skip if threat intel service is not available
      const serviceAvailable = await isThreatIntelAvailable();
      if (!serviceAvailable) {
        console.warn('Skipping test - threat intel service not available');
        return;
      }

      const reportData = {
        url: 'https://malicious.com',
        reason: 'Phishing attempt',
        description: 'Suspicious website',
        threatType: 'phishing',
        severity: 'high' as const,
        confidence: 90,
      };

      const response = await request(app)
        .post('/api/v1/extension/report')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send(reportData);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should report email threat successfully', async () => {
      const serviceAvailable = await isThreatIntelAvailable();
      if (!serviceAvailable) {
        console.warn('Skipping test - threat intel service not available');
        return;
      }

      const reportData = {
        email: 'phishing@malicious.com',
        reason: 'Phishing email',
        threatType: 'phishing',
        severity: 'medium' as const,
      };

      const response = await request(app)
        .post('/api/v1/extension/report')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send(reportData);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should report domain threat successfully', async () => {
      const serviceAvailable = await isThreatIntelAvailable();
      if (!serviceAvailable) {
        console.warn('Skipping test - threat intel service not available');
        return;
      }

      const reportData = {
        domain: 'malicious.com',
        reason: 'Malicious domain',
      };

      const response = await request(app)
        .post('/api/v1/extension/report')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send(reportData);

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should require at least one of url, email, or domain', async () => {
      const response = await request(app)
        .post('/api/v1/extension/report')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({
          reason: 'Test reason',
        });

      expect(response.status).toBe(400);
      expect(response.body.error.message).toContain('url, email, or domain is required');
    });

    it('should require reason', async () => {
      const response = await request(app)
        .post('/api/v1/extension/report')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({
          url: 'https://example.com',
        });

      expect(response.status).toBe(400);
      expect(response.body.error.message).toBe('Validation failed');
    });

    it('should validate URL format', async () => {
      const response = await request(app)
        .post('/api/v1/extension/report')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({
          url: 'not-a-valid-url',
          reason: 'Test',
        });

      expect(response.status).toBe(400);
    });

    it('should validate email format', async () => {
      const response = await request(app)
        .post('/api/v1/extension/report')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({
          email: 'not-an-email',
          reason: 'Test',
        });

      expect(response.status).toBe(400);
    });

    it('should validate severity enum', async () => {
      const response = await request(app)
        .post('/api/v1/extension/report')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({
          url: 'https://example.com',
          reason: 'Test',
          severity: 'invalid',
        });

      expect(response.status).toBe(400);
    });

    it('should handle threat intelligence service errors gracefully', async () => {
      // This will fail gracefully even if service is unavailable
      const reportData = {
        url: 'https://example.com',
        reason: 'Test',
      };

      const response = await request(app)
        .post('/api/v1/extension/report')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send(reportData);

      // Should return success even if service unavailable (per design)
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
  });
});
