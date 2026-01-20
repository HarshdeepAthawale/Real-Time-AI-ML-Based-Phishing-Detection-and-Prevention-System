import { CacheService } from '../../../src/services/cache.service';
import Redis from 'ioredis';
import crypto from 'crypto';

jest.mock('ioredis');
jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('CacheService', () => {
  let cacheService: CacheService;
  let mockRedis: jest.Mocked<Redis>;

  beforeEach(() => {
    jest.clearAllMocks();
    mockRedis = {
      get: jest.fn(),
      setex: jest.fn(),
      ping: jest.fn(),
      quit: jest.fn(),
      on: jest.fn(),
    } as any;

    (Redis as jest.MockedClass<typeof Redis>).mockImplementation(() => mockRedis);
    cacheService = new CacheService();
  });

  describe('get', () => {
    it('should retrieve cached value', async () => {
      const cachedValue = { test: 'data' };
      mockRedis.get.mockResolvedValue(JSON.stringify(cachedValue));

      const result = await cacheService.get('test-key');

      expect(result).toEqual(cachedValue);
      expect(mockRedis.get).toHaveBeenCalledWith('test-key');
    });

    it('should return null when key does not exist', async () => {
      mockRedis.get.mockResolvedValue(null);

      const result = await cacheService.get('non-existent-key');

      expect(result).toBeNull();
    });

    it('should handle JSON parse errors gracefully', async () => {
      mockRedis.get.mockResolvedValue('invalid-json');

      const result = await cacheService.get('invalid-key');

      expect(result).toBeNull();
    });

    it('should handle Redis errors gracefully', async () => {
      mockRedis.get.mockRejectedValue(new Error('Redis error'));

      const result = await cacheService.get('error-key');

      expect(result).toBeNull();
    });
  });

  describe('set', () => {
    it('should set value with TTL', async () => {
      const value = { test: 'data' };
      mockRedis.setex.mockResolvedValue('OK');

      await cacheService.set('test-key', value, 3600);

      expect(mockRedis.setex).toHaveBeenCalledWith(
        'test-key',
        3600,
        JSON.stringify(value)
      );
    });

    it('should use default TTL when not specified', async () => {
      const value = { test: 'data' };
      mockRedis.setex.mockResolvedValue('OK');

      await cacheService.set('test-key', value);

      expect(mockRedis.setex).toHaveBeenCalledWith(
        'test-key',
        3600,
        JSON.stringify(value)
      );
    });

    it('should handle Redis errors gracefully', async () => {
      mockRedis.setex.mockRejectedValue(new Error('Redis error'));

      await expect(
        cacheService.set('error-key', { test: 'data' })
      ).resolves.not.toThrow();
    });
  });

  describe('getOrSet', () => {
    it('should return cached value when available', async () => {
      const cachedValue = { test: 'cached' };
      mockRedis.get.mockResolvedValue(JSON.stringify(cachedValue));

      const fetcher = jest.fn().mockResolvedValue({ test: 'new' });

      const result = await cacheService.getOrSet('test-key', fetcher);

      expect(result).toEqual(cachedValue);
      expect(fetcher).not.toHaveBeenCalled();
    });

    it('should call fetcher and cache result when not cached', async () => {
      mockRedis.get.mockResolvedValue(null);
      mockRedis.setex.mockResolvedValue('OK');

      const newValue = { test: 'new' };
      const fetcher = jest.fn().mockResolvedValue(newValue);

      const result = await cacheService.getOrSet('test-key', fetcher, 7200);

      expect(result).toEqual(newValue);
      expect(fetcher).toHaveBeenCalled();
      expect(mockRedis.setex).toHaveBeenCalledWith(
        'test-key',
        7200,
        JSON.stringify(newValue)
      );
    });
  });

  describe('generateCacheKey', () => {
    it('should generate consistent cache keys', () => {
      const input = 'test-input';
      const key1 = cacheService.generateCacheKey('email', input);
      const key2 = cacheService.generateCacheKey('email', input);

      expect(key1).toBe(key2);
      expect(key1).toContain('detection:email:');
    });

    it('should generate different keys for different inputs', () => {
      const key1 = cacheService.generateCacheKey('email', 'input1');
      const key2 = cacheService.generateCacheKey('email', 'input2');

      expect(key1).not.toBe(key2);
    });

    it('should generate different keys for different types', () => {
      const key1 = cacheService.generateCacheKey('email', 'same-input');
      const key2 = cacheService.generateCacheKey('url', 'same-input');

      expect(key1).not.toBe(key2);
    });

    it('should use SHA-256 hashing', () => {
      const input = 'test-input';
      const key = cacheService.generateCacheKey('email', input);
      const hash = crypto.createHash('sha256').update(input).digest('hex');

      expect(key).toContain(hash);
    });
  });

  describe('isConnected', () => {
    it('should return true when Redis is connected', async () => {
      mockRedis.ping.mockResolvedValue('PONG');

      const result = await cacheService.isConnected();

      expect(result).toBe(true);
      expect(mockRedis.ping).toHaveBeenCalled();
    });

    it('should return false when Redis is not connected', async () => {
      mockRedis.ping.mockRejectedValue(new Error('Connection failed'));

      const result = await cacheService.isConnected();

      expect(result).toBe(false);
    });
  });

  describe('disconnect', () => {
    it('should disconnect from Redis', async () => {
      mockRedis.quit.mockResolvedValue('OK');

      await cacheService.disconnect();

      expect(mockRedis.quit).toHaveBeenCalled();
    });
  });
});
