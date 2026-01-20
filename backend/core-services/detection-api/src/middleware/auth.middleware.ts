import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';

export interface AuthenticatedRequest extends Request {
  apiKey?: string;
  apiKeyId?: string;
  organizationId?: string;
}

export const authMiddleware = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): void => {
  try {
    const apiKey = req.headers['x-api-key'] as string;

    if (!apiKey) {
      logger.warn('API key missing', { path: req.path });
      return res.status(401).json({
        error: {
          message: 'API key required',
          statusCode: 401
        }
      });
    }

    // TODO: Validate API key against database
    // For now, we'll just attach it to the request
    req.apiKey = apiKey;
    req.apiKeyId = 'temp-id'; // Replace with actual validation
    req.organizationId = req.headers['x-organization-id'] as string;

    logger.debug('API key authenticated', {
      apiKeyId: req.apiKeyId,
      path: req.path
    });

    next();
  } catch (error) {
    logger.error('API key authentication failed', { error });
    return res.status(401).json({
      error: {
        message: 'Invalid API key',
        statusCode: 401
      }
    });
  }
};
