import { Request, Response, NextFunction } from 'express';
import rateLimit from 'express-rate-limit';
import { rateLimitConfig } from '../config/gateway';

export const rateLimiter = rateLimit({
  ...rateLimitConfig,
  keyGenerator: (req: Request) => {
    // Use API key for rate limiting if available
    const apiKey = req.headers['x-api-key'] as string;
    return apiKey || req.ip;
  },
  handler: (req: Request, res: Response) => {
    res.status(429).json({
      error: {
        message: 'Too many requests, please try again later',
        statusCode: 429,
        retryAfter: Math.round(rateLimitConfig.windowMs / 1000)
      }
    });
  }
});
