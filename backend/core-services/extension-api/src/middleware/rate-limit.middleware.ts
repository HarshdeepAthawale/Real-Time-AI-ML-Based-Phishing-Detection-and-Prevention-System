import { Request, Response, NextFunction } from 'express';
import { ExtensionRequest } from './extension-auth.middleware';
import { CacheService } from '../services/cache.service';
import { config } from '../config';
import { logger } from '../utils/logger';
import crypto from 'crypto';

/**
 * Rate limiting middleware for extension endpoints
 * Uses Redis to track request counts per extension ID
 */
export const createRateLimitMiddleware = (cacheService: CacheService) => {
  return async (
    req: ExtensionRequest,
    res: Response,
    next: NextFunction
  ): Promise<void> => {
    try {
      const extensionId = req.extensionId || req.ip || 'unknown';
      const windowMs = config.rateLimit.windowMs;
      const maxRequests = config.rateLimit.extensionMax;
      
      // Generate rate limit key
      const rateLimitKey = `extension:ratelimit:${crypto.createHash('sha256').update(extensionId).digest('hex').substring(0, 16)}`;
      
      // Check current count
      const currentCount = await cacheService.get(rateLimitKey);
      const count = currentCount ? parseInt(currentCount.toString(), 10) : 0;
      
      if (count >= maxRequests) {
        logger.warn('Rate limit exceeded', {
          extensionId,
          count,
          maxRequests
        });
        
        res.status(429).json({
          error: {
            message: 'Rate limit exceeded',
            statusCode: 429,
            retryAfter: Math.ceil(windowMs / 1000) // seconds
          }
        });
        return;
      }
      
      // Increment count
      const ttlSeconds = Math.ceil(windowMs / 1000);
      await cacheService.set(rateLimitKey, count + 1, ttlSeconds);
      
      // Add rate limit headers
      res.setHeader('X-RateLimit-Limit', maxRequests.toString());
      res.setHeader('X-RateLimit-Remaining', Math.max(0, maxRequests - count - 1).toString());
      res.setHeader('X-RateLimit-Reset', new Date(Date.now() + windowMs).toISOString());
      
      next();
    } catch (error: any) {
      // On error, allow request but log
      logger.error('Rate limit check failed', { error: error.message });
      // Continue without rate limiting if Redis is unavailable
      next();
    }
  };
};
