import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';
import { getPostgreSQL } from '../../../../shared/database/connection';
import { ApiKey } from '../../../../shared/database/models/ApiKey';
import { verifyApiKey } from '../../../../shared/utils/apiKeyManager';

export interface AuthenticatedRequest extends Request {
  apiKey?: string;
  apiKeyId?: string;
  organizationId?: string;
}

export const authMiddleware = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const apiKey = req.headers['x-api-key'] as string;

    if (!apiKey) {
      logger.warn('API key missing', { path: req.path });
      res.status(401).json({
        error: {
          message: 'API key required',
          statusCode: 401
        }
      });
      return;
    }

    // Validate API key against database
    try {
      const dataSource = getPostgreSQL();
      const apiKeyRepository = dataSource.getRepository(ApiKey);
      
      // Get all active API keys (not revoked, not expired)
      const now = new Date();
      const activeApiKeys = await apiKeyRepository.find({
        where: {
          revoked_at: null as any, // TypeORM null check
        },
        relations: ['organization'],
      });

      // Filter out expired keys and verify against each hash
      let validApiKey: ApiKey | null = null;
      for (const key of activeApiKeys) {
        // Check if expired
        if (key.expires_at && new Date(key.expires_at) < now) {
          continue;
        }

        // Verify API key against hash
        const isValid = await verifyApiKey(apiKey, key.key_hash);
        if (isValid) {
          validApiKey = key;
          break;
        }
      }

      if (!validApiKey) {
        logger.warn('Invalid API key', { path: req.path });
        res.status(401).json({
          error: {
            message: 'Invalid API key',
            statusCode: 401
          }
        });
        return;
      }

      // Update last_used_at
      validApiKey.last_used_at = new Date();
      await apiKeyRepository.save(validApiKey);

      // Attach validated data to request
      req.apiKey = apiKey;
      req.apiKeyId = validApiKey.id;
      req.organizationId = validApiKey.organization_id;

      logger.debug('API key authenticated', {
        apiKeyId: req.apiKeyId,
        organizationId: req.organizationId,
        path: req.path
      });

      next();
    } catch (dbError) {
      logger.error('Database error during API key validation', { error: dbError });
      // In development, allow fallback; in production, reject
      if (process.env.NODE_ENV === 'development') {
        logger.warn('Falling back to basic auth in development mode');
        req.apiKey = apiKey;
        req.apiKeyId = 'dev-mode';
        req.organizationId = req.headers['x-organization-id'] as string;
        next();
      } else {
        res.status(500).json({
          error: {
            message: 'Authentication service unavailable',
            statusCode: 500
          }
        });
      }
    }
  } catch (error) {
    logger.error('API key authentication failed', { error });
    res.status(401).json({
      error: {
        message: 'Invalid API key',
        statusCode: 401
      }
    });
  }
};
