import { Request, Response } from 'express';
import rateLimit from 'express-rate-limit';
import { config } from '../config';

export const rateLimitMiddleware = rateLimit({
  windowMs: config.rateLimit.windowMs,
  max: config.rateLimit.max,
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
        retryAfter: Math.round(config.rateLimit.windowMs / 1000)
      }
    });
  }
});
