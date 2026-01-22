import { Request, Response } from 'express';
import rateLimit from 'express-rate-limit';
import { config } from '../config';

// Rate limit configuration - same as detection-api
const rateLimitConfig = {
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // requests per window
};

export const rateLimitMiddleware = rateLimit({
  windowMs: rateLimitConfig.windowMs,
  max: rateLimitConfig.max,
  standardHeaders: true,
  legacyHeaders: false,
  keyGenerator: (req: Request) => {
    // Use API key for rate limiting if available
    const apiKey = req.headers['x-api-key'] as string;
    return apiKey || req.ip || 'unknown';
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
