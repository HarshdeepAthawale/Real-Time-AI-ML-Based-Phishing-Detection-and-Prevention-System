import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';

export interface AuthenticatedRequest extends Request {
  apiKey?: string;
  apiKeyId?: string;
}

export const apiKeyAuth = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const apiKey = req.headers['x-api-key'] as string;

    if (!apiKey) {
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
