import request from 'supertest';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { EmailClientService } from '../../../src/services/email-client.service';
import { EmailScannerService } from '../../../src/services/email-scanner.service';
import { PrivacyFilterService } from '../../../src/services/privacy-filter.service';
import { URLCheckerService } from '../../../src/services/url-checker.service';
import { CacheService } from '../../../src/services/cache.service';
import { extensionAuthMiddleware } from '../../../src/middleware/extension-auth.middleware';
import { createRateLimitMiddleware } from '../../../src/middleware/rate-limit.middleware';
import emailClientRoutes from '../../../src/routes/email-client.routes';
import { connectTestDatabase, truncateTables } from '../../helpers/test-db';
import { flushTestRedis, getTestRedisUrl } from '../../helpers/test-redis';
import { createStandardTestFixtures, TEST_API_KEY } from '../../fixtures/api-keys';

describe('Email Client Route Integration', () => {
  let app: express.Application;
  let emailClient: EmailClientService;
  let emailScanner: EmailScannerService;
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
    
    // Create real service instances
    cacheService = new CacheService();
    const privacyFilter = new PrivacyFilterService();
    
    // Wait for cache service to connect
    let retries = 10;
    while (retries > 0 && !(await cacheService.isConnected())) {
      await new Promise(resolve => setTimeout(resolve, 100));
      retries--;
    }
    
    const urlChecker = new URLCheckerService(
      process.env.DETECTION_API_URL || 'http://localhost:3001',
      cacheService,
      privacyFilter
    );
    
    emailScanner = new EmailScannerService(
      process.env.DETECTION_API_URL || 'http://localhost:3001',
      privacyFilter,
      urlChecker
    );
    
    emailClient = new EmailClientService();

    // Create Express app
    app = express();
    app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
    app.use(cors({ origin: true, credentials: true }));
    app.use(express.json({ limit: '1mb' }));
    app.use(express.urlencoded({ extended: true }));

    app.set('emailClient', emailClient);
    app.set('emailScanner', emailScanner);
    app.set('cacheService', cacheService);

    app.use(
      '/api/v1/extension/email',
      extensionAuthMiddleware,
      createRateLimitMiddleware(cacheService),
      emailClientRoutes
    );
  });

  afterEach(async () => {
    try {
      if (emailClient) {
        await emailClient.disconnectAll();
      }
    } catch {
      // Ignore errors
    }
    try {
      if (cacheService) {
        await cacheService.disconnect();
      }
    } catch {
      // Ignore disconnect errors
    }
    await flushTestRedis();
  });

  describe('POST /api/v1/extension/email/connect', () => {
    it('should validate required fields', async () => {
      const response = await request(app)
        .post('/api/v1/extension/email/connect')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({
          host: 'imap.example.com',
          // Missing required fields
        });

      expect(response.status).toBe(400);
      expect(response.body.error.message).toBe('Validation failed');
    });

    it('should validate email format for user', async () => {
      const response = await request(app)
        .post('/api/v1/extension/email/connect')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({
          id: 'account-1',
          host: 'imap.example.com',
          port: 993,
          user: 'not-an-email',
          password: 'password',
        });

      expect(response.status).toBe(400);
    });

    it('should validate port range', async () => {
      const response = await request(app)
        .post('/api/v1/extension/email/connect')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({
          id: 'account-1',
          host: 'imap.example.com',
          port: 70000, // Invalid port
          user: 'test@example.com',
          password: 'password',
        });

      expect(response.status).toBe(400);
    });

    // Note: Actual IMAP connection tests require a test IMAP server
    // These are skipped as they need real email server infrastructure
  });

  describe('POST /api/v1/extension/email/disconnect', () => {
    it('should require accountId', async () => {
      const response = await request(app)
        .post('/api/v1/extension/email/disconnect')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({});

      expect(response.status).toBe(400);
      expect(response.body.error.message).toBe('Account ID is required');
    });
  });

  describe('GET /api/v1/extension/email/status', () => {
    it('should return status for all accounts', async () => {
      const response = await request(app)
        .get('/api/v1/extension/email/status')
        .set('X-API-Key', standardFixtures.fullApiKey);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('accounts');
      expect(response.body).toHaveProperty('total');
    });
  });

  describe('POST /api/v1/extension/email/scan', () => {
    it('should require accountId', async () => {
      const response = await request(app)
        .post('/api/v1/extension/email/scan')
        .set('X-API-Key', standardFixtures.fullApiKey)
        .send({});

      expect(response.status).toBe(400);
      expect(response.body.error.message).toBe('Account ID is required');
    });
  });
});
