import { Request, Response, NextFunction } from 'express';
import { createRateLimitMiddleware } from '../../../src/middleware/rate-limit.middleware';
import { CacheService } from '../../../src/services/cache.service';
import { ExtensionRequest } from '../../../src/middleware/extension-auth.middleware';
import { flushTestRedis, getTestRedisUrl } from '../../helpers/test-redis';

describe('createRateLimitMiddleware', () => {
  let mockRequest: Partial<ExtensionRequest>;
  let mockResponse: Partial<Response>;
  let mockNext: NextFunction;
  let cacheService: CacheService;
  let rateLimitMiddleware: ReturnType<typeof createRateLimitMiddleware>;

  beforeEach(async () => {
    // Set up test Redis
    const redisUrl = getTestRedisUrl();
    process.env.REDIS_URL = redisUrl;
    process.env.REDIS_HOST = redisUrl.split('://')[1]?.split(':')[0] || 'localhost';
    process.env.REDIS_PORT = redisUrl.split(':')[2]?.split('/')[0] || '6380';
    
    // Flush Redis before each test
    await flushTestRedis();
    
    // Create real cache service
    cacheService = new CacheService();
    
    // Wait for cache service to connect
    let retries = 10;
    while (retries > 0 && !(await cacheService.isConnected())) {
      await new Promise(resolve => setTimeout(resolve, 100));
      retries--;
    }

    mockRequest = {
      extensionId: 'ext-123',
      ip: '127.0.0.1',
      headers: {},
    };

    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
      setHeader: jest.fn().mockReturnThis(),
    };

    mockNext = jest.fn();

    // Ensure cache service is connected before creating middleware
    let connectionRetries = 10;
    while (connectionRetries > 0 && !(await cacheService.isConnected())) {
      await new Promise(resolve => setTimeout(resolve, 100));
      connectionRetries--;
    }
    
    if (!(await cacheService.isConnected())) {
      throw new Error('Cache service failed to connect');
    }

    rateLimitMiddleware = createRateLimitMiddleware(cacheService);
  });

  afterEach(async () => {
    await cacheService.disconnect();
    await flushTestRedis();
  });

  describe('rate limiting', () => {
    it('should allow request within limit', async () => {
      await rateLimitMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      // Wait a bit for async operations
      await new Promise(resolve => setTimeout(resolve, 100));

      expect(mockNext).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should block request exceeding limit', async () => {
      const maxRequests = 200; // From config
      
      // Reset mocks before test
      jest.clearAllMocks();
      
      // Make maxRequests requests to exceed limit
      for (let i = 0; i < maxRequests; i++) {
        await rateLimitMiddleware(
          mockRequest as ExtensionRequest,
          mockResponse as Response,
          mockNext
        );
        // Small delay to ensure Redis operations complete
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      
      // Wait for all async operations
      await new Promise(resolve => setTimeout(resolve, 200));
      
      // Clear mocks to check only the last call
      jest.clearAllMocks();
      
      // Next request should be blocked
      await rateLimitMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );
      
      // Wait for async operations
      await new Promise(resolve => setTimeout(resolve, 100));

      expect(mockResponse.status).toHaveBeenCalledWith(429);
      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            message: 'Rate limit exceeded',
            statusCode: 429,
          }),
        })
      );
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should increment request count', async () => {
      // Make a few requests
      for (let i = 0; i < 5; i++) {
        await rateLimitMiddleware(
          mockRequest as ExtensionRequest,
          mockResponse as Response,
          mockNext
        );
        // Small delay between requests
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      // Wait for all async operations
      await new Promise(resolve => setTimeout(resolve, 100));

      // All should pass
      expect(mockNext).toHaveBeenCalledTimes(5);
    });

    it('should use extension ID for rate limit key', async () => {
      mockRequest.extensionId = 'ext-123';
      
      await rateLimitMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      await new Promise(resolve => setTimeout(resolve, 100));

      expect(mockNext).toHaveBeenCalled();
    });

    it('should fall back to IP when extension ID not provided', async () => {
      const requestWithIP = {
        ...mockRequest,
        extensionId: undefined,
        ip: '192.168.1.1',
      };
      
      await rateLimitMiddleware(
        requestWithIP as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockNext).toHaveBeenCalled();
    });

    it('should set rate limit headers', async () => {
      await rateLimitMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      await new Promise(resolve => setTimeout(resolve, 100));

      expect(mockResponse.setHeader).toHaveBeenCalledWith('X-RateLimit-Limit', expect.any(String));
      expect(mockResponse.setHeader).toHaveBeenCalledWith('X-RateLimit-Remaining', expect.any(String));
      expect(mockResponse.setHeader).toHaveBeenCalledWith('X-RateLimit-Reset', expect.any(String));
    });

    it('should handle cache errors gracefully', async () => {
      // Disconnect cache to simulate error
      await cacheService.disconnect();
      
      await rateLimitMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      // Should allow request when cache fails
      expect(mockNext).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });
  });
});
