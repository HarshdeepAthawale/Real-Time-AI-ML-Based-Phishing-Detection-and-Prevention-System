import { Request, Response } from 'express';
import { rateLimitMiddleware } from '../../../src/middleware/rate-limit.middleware';

// Mock express-rate-limit
jest.mock('express-rate-limit', () => {
  return jest.fn(() => (req: Request, res: Response, next: any) => {
    // Simple mock that allows requests
    next();
  });
});

describe('rateLimitMiddleware', () => {
  it('should be defined', () => {
    expect(rateLimitMiddleware).toBeDefined();
  });

  it('should be a function', () => {
    expect(typeof rateLimitMiddleware).toBe('function');
  });

  // Note: Full rate limiting tests would require more complex mocking
  // of express-rate-limit internals. The middleware itself is a thin
  // wrapper around express-rate-limit, so integration tests are more
  // appropriate for testing actual rate limiting behavior.
});
