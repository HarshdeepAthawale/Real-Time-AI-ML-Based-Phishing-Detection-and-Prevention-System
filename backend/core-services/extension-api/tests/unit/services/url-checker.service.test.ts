import { URLCheckerService } from '../../../src/services/url-checker.service';
import { CacheService } from '../../../src/services/cache.service';
import { PrivacyFilterService } from '../../../src/services/privacy-filter.service';
import { flushTestRedis, getTestRedisUrl } from '../../helpers/test-redis';
import { isDetectionApiAvailable, skipIfServiceUnavailable } from '../../helpers/test-services';

describe('URLCheckerService', () => {
  let urlChecker: URLCheckerService;
  let cacheService: CacheService;
  let privacyFilter: PrivacyFilterService;
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
  });

  afterEach(async () => {
    await cacheService.disconnect();
    await flushTestRedis();
  });

  describe('checkURL', () => {
    const testUrl = 'https://example.com/test';

    it('should return cached result when cache hit', async () => {
      const cachedResult = {
        isThreat: false,
        severity: 'low',
        confidence: 20,
        warningMessage: undefined,
        details: {},
      };

      // Set cache first
      const cacheKey = cacheService.generateURLKey(testUrl);
      await cacheService.set(cacheKey, cachedResult, 60);

      const result = await urlChecker.checkURL(testUrl);

      expect(result.isThreat).toBe(false);
      expect(result.severity).toBe('low');
      expect(result.confidence).toBe(20);
      expect(result.cached).toBe(true);
    });

    it('should call detection API when cache miss', async () => {
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await urlChecker.checkURL(testUrl);

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('cached');
      expect(result.cached).toBe(false);
      expect(result.timestamp).toBeDefined();
    });

    it('should use privacy filter when privacyMode is enabled', async () => {
      const filteredUrl = privacyFilter.filterURL(testUrl);
      expect(filteredUrl).toBe('https://example.com');
      
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await urlChecker.checkURL(testUrl, { privacyMode: true });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('cached');
    });

    it('should include pageText when provided', async () => {
      const pageText = 'Sample page text';
      
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await urlChecker.checkURL(testUrl, { pageText });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('cached');
    });

    it('should include screenshot when provided', async () => {
      const screenshot = 'base64encodedimage';
      
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await urlChecker.checkURL(testUrl, { screenshot });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('cached');
    });

    it('should return safe default on error', async () => {
      // Use invalid URL to trigger error
      const invalidUrl = 'not-a-valid-url';
      
      const result = await urlChecker.checkURL(invalidUrl);

      expect(result.isThreat).toBe(false);
      expect(result.cached).toBe(false);
      expect(result.timestamp).toBeDefined();
    });

    it('should use privacy filter for pageText when privacyMode is enabled', async () => {
      const pageText = 'Full page content';
      const filteredText = privacyFilter.extractTextContent(pageText);
      
      expect(filteredText).toBeTruthy();
      
      // Skip if detection API is not available
      const apiAvailable = await isDetectionApiAvailable(detectionApiUrl);
      if (!apiAvailable) {
        console.warn('Skipping test - detection API not available');
        return;
      }

      const result = await urlChecker.checkURL(testUrl, { pageText, privacyMode: true });

      expect(result).toHaveProperty('isThreat');
      expect(result).toHaveProperty('cached');
    });
  });
});
