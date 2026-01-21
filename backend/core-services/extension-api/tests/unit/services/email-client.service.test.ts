import { EmailClientService } from '../../../src/services/email-client.service';
import { EmailScannerService } from '../../../src/services/email-scanner.service';
import { PrivacyFilterService } from '../../../src/services/privacy-filter.service';
import { URLCheckerService } from '../../../src/services/url-checker.service';
import { CacheService } from '../../../src/services/cache.service';
import { flushTestRedis, getTestRedisUrl } from '../../helpers/test-redis';

describe('EmailClientService', () => {
  let emailClient: EmailClientService;
  let scanner: EmailScannerService;
  let cacheService: CacheService;

  beforeEach(async () => {
    // Set up test Redis
    const redisUrl = getTestRedisUrl();
    process.env.REDIS_URL = redisUrl;
    process.env.REDIS_HOST = redisUrl.split('://')[1]?.split(':')[0] || 'localhost';
    process.env.REDIS_PORT = redisUrl.split(':')[2]?.split('/')[0] || '6380';
    
    await flushTestRedis();
    
    cacheService = new CacheService();
    let retries = 10;
    while (retries > 0 && !(await cacheService.isConnected())) {
      await new Promise(resolve => setTimeout(resolve, 100));
      retries--;
    }
    
    const privacyFilter = new PrivacyFilterService();
    const urlChecker = new URLCheckerService(
      process.env.DETECTION_API_URL || 'http://localhost:3001',
      cacheService,
      privacyFilter
    );
    scanner = new EmailScannerService(
      process.env.DETECTION_API_URL || 'http://localhost:3001',
      privacyFilter,
      urlChecker
    );

    emailClient = new EmailClientService();
  });

  afterEach(async () => {
    try {
      await emailClient.disconnectAll();
    } catch {
      // Ignore disconnect errors
    }
    await cacheService.disconnect();
    await flushTestRedis();
  });

  describe('setScanner', () => {
    it('should set scanner for account', () => {
      emailClient.setScanner('test-account', scanner);
      const status = emailClient.getStatus('test-account');
      expect(status).toBeDefined();
    });
  });

  describe('getStatus', () => {
    it('should return connection status', () => {
      const status = emailClient.getStatus('non-existent');
      expect(status).toEqual({
        connected: false,
        scanning: false,
      });
    });
  });

  describe('getConnectedAccounts', () => {
    it('should return empty array when no accounts connected', () => {
      const accounts = emailClient.getConnectedAccounts();
      expect(accounts).toEqual([]);
    });
  });

  describe('disconnect', () => {
    it('should handle disconnect for non-existent account gracefully', async () => {
      await expect(emailClient.disconnect('non-existent')).resolves.not.toThrow();
    });
  });

  describe('disconnectAll', () => {
    it('should disconnect all accounts', async () => {
      await expect(emailClient.disconnectAll()).resolves.not.toThrow();
    });
  });

  // Note: IMAP connection tests are skipped as they require a real IMAP server
  // These would be better suited as integration tests with a test IMAP server
  describe('IMAP connection (requires test server)', () => {
    it.skip('should connect to email account when IMAP server available', async () => {
      // This test requires a test IMAP server
      // Can be enabled when test infrastructure includes IMAP server
    });
  });
});
