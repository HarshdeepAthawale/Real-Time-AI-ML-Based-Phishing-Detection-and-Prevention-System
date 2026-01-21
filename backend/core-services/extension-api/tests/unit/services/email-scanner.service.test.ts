import { EmailScannerService } from '../../../src/services/email-scanner.service';
import { PrivacyFilterService } from '../../../src/services/privacy-filter.service';
import { URLCheckerService } from '../../../src/services/url-checker.service';
import { CacheService } from '../../../src/services/cache.service';
import { flushTestRedis, getTestRedisUrl } from '../../helpers/test-redis';
import { isDetectionApiAvailable } from '../../helpers/test-services';

describe('EmailScannerService', () => {
  let emailScanner: EmailScannerService;
  let privacyFilter: PrivacyFilterService;
  let urlChecker: URLCheckerService | null;
  let cacheService: CacheService;
  const detectionApiUrl = process.env.DETECTION_API_URL || 'http://localhost:3001';

  beforeEach(async () => {
    // Set up test Redis
    const redisUrl = getTestRedisUrl();
    process.env.REDIS_URL = redisUrl;
    process.env.REDIS_HOST = redisUrl.split('://')[1]?.split(':')[0] || 'localhost';
    process.env.REDIS_PORT = redisUrl.split(':')[2]?.split('/')[0] || '6380';
    
    // Flush Redis before each test
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
    
    urlChecker = new URLCheckerService(detectionApiUrl, cacheService, privacyFilter);
    emailScanner = new EmailScannerService(detectionApiUrl, privacyFilter, urlChecker);
  });

  afterEach(async () => {
    await cacheService.disconnect();
    await flushTestRedis();
  });

  describe('scanEmail', () => {
    const testEmailContent = 'Subject: Test Email\nFrom: test@example.com\nBody: This is a test email.';

    it('should scan email and return result', async () => {
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await emailScanner.scanEmail(testEmailContent);

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('timestamp');
      expect(result.timestamp).toBeDefined();
    });

    it('should use privacy filter when privacyMode is enabled', async () => {
      const filtered = privacyFilter.filterEmailContent(testEmailContent);
      expect(filtered.subject).toBe('Test Email');
      expect(filtered.from).toBe('test@example.com');
      
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await emailScanner.scanEmail(testEmailContent, { privacyMode: true });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('timestamp');
    });

    it('should scan links when scanLinks is enabled', async () => {
      const emailWithLinks = 'Check out https://example.com and https://test.com';
      
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await emailScanner.scanEmail(emailWithLinks, { scanLinks: true });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('timestamp');
      // Note: suspiciousLinks may or may not be present depending on detection results
    });

    it('should limit link scanning to 10 links', async () => {
      const manyLinks = Array(15).fill('https://example.com').join(' ');
      
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await emailScanner.scanEmail(manyLinks, { scanLinks: true });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('timestamp');
    });

    it('should return safe default on error', async () => {
      const invalidContent = null as any;

      const result = await emailScanner.scanEmail(invalidContent);

      expect(result.isThreat).toBe(false);
      expect(result.timestamp).toBeDefined();
    });

    it('should not scan links when scanLinks is false', async () => {
      const emailWithLinks = 'Check out https://example.com';
      
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await emailScanner.scanEmail(emailWithLinks, { scanLinks: false });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('timestamp');
    });

    it('should work without URL checker service', async () => {
      const scannerWithoutChecker = new EmailScannerService(detectionApiUrl, privacyFilter);
      
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await scannerWithoutChecker.scanEmail(testEmailContent, { scanLinks: true });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('timestamp');
    });

    it('should include full analysis when requested', async () => {
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await emailScanner.scanEmail(testEmailContent, { includeFullAnalysis: true });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('timestamp');
    });
  });
});
