import { Request, Response, NextFunction } from 'express';
import bcrypt from 'bcrypt';
import { logger } from '../utils/logger';
// @ts-ignore - Module resolved by Jest moduleNameMapper at runtime
import { getPostgreSQL } from '../../shared/database';
// @ts-ignore - Module resolved by Jest moduleNameMapper at runtime
import { ApiKey } from '../../shared/database/models';

export interface ExtensionRequest extends Request {
  extensionId?: string;
  apiKey?: string;
  organizationId?: string;
  apiKeyEntity?: ApiKey;
}

/**
 * Middleware to authenticate browser extensions
 * Supports API key authentication via X-API-Key header
 * Validates API key against database (key_hash, expiration, revocation)
 * Optionally validates extension ID via X-Extension-Id header
 */
export const extensionAuthMiddleware = async (
  req: ExtensionRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const apiKey = req.headers['x-api-key'] as string;
    const extensionId = req.headers['x-extension-id'] as string;
    const organizationId = req.headers['x-organization-id'] as string;
    
    // API key is required
    if (!apiKey) {
      logger.warn('Extension API request missing API key', {
        path: req.path,
        ip: req.ip
      });
      res.status(401).json({
        error: {
          message: 'API key required',
          statusCode: 401
        }
      });
      return;
    }
    
    // Validate API key format (should have prefix format: prefix_xxxxx)
    if (apiKey.length < 10 || !apiKey.includes('_')) {
      logger.warn('Invalid API key format', {
        path: req.path,
        ip: req.ip,
        keyLength: apiKey.length
      });
      res.status(401).json({
        error: {
          message: 'Invalid API key format',
          statusCode: 401
        }
      });
      return;
    }
    
    // Extract key prefix (first part before underscore)
    const keyPrefix = apiKey.split('_')[0];
    
    try {
      // Get database connection
      const dataSource = getPostgreSQL();
      const apiKeyRepository = dataSource.getRepository(ApiKey);
      
      // Find API key by prefix
      const apiKeyEntity = await apiKeyRepository.findOne({
        where: { key_prefix: keyPrefix },
        relations: ['organization']
      });
      
      if (!apiKeyEntity) {
        logger.warn('API key not found', {
          prefix: keyPrefix,
          path: req.path,
          ip: req.ip
        });
        res.status(401).json({
          error: {
            message: 'Invalid API key',
            statusCode: 401
          }
        });
        return;
      }
      
      // Verify key hash using bcrypt
      const isValidKey = await bcrypt.compare(apiKey, apiKeyEntity.key_hash);
      
      if (!isValidKey) {
        logger.warn('API key hash mismatch', {
          prefix: keyPrefix,
          path: req.path,
          ip: req.ip
        });
        res.status(401).json({
          error: {
            message: 'Invalid API key',
            statusCode: 401
          }
        });
        return;
      }
      
      // Check if key is revoked
      if (apiKeyEntity.revoked_at) {
        logger.warn('API key revoked', {
          prefix: keyPrefix,
          revokedAt: apiKeyEntity.revoked_at,
          path: req.path,
          ip: req.ip
        });
        res.status(401).json({
          error: {
            message: 'API key has been revoked',
            statusCode: 401
          }
        });
        return;
      }
      
      // Check if key is expired
      if (apiKeyEntity.expires_at && new Date(apiKeyEntity.expires_at) < new Date()) {
        logger.warn('API key expired', {
          prefix: keyPrefix,
          expiresAt: apiKeyEntity.expires_at,
          path: req.path,
          ip: req.ip
        });
        res.status(401).json({
          error: {
            message: 'API key has expired',
            statusCode: 401
          }
        });
        return;
      }
      
      // Update last_used_at timestamp
      apiKeyEntity.last_used_at = new Date();
      await apiKeyRepository.save(apiKeyEntity);
      
      // Attach to request for use in routes
      req.apiKey = apiKey;
      req.extensionId = extensionId;
      req.organizationId = organizationId || apiKeyEntity.organization_id;
      req.apiKeyEntity = apiKeyEntity;
      
      logger.debug('Extension authenticated', {
        extensionId: extensionId || 'unknown',
        organizationId: req.organizationId,
        path: req.path,
        apiKeyPrefix: keyPrefix
      });
      
      next();
    } catch (dbError: any) {
      // Fail fast if database is unavailable (per requirement)
      logger.error('Database error during API key validation', {
        error: dbError.message,
        path: req.path,
        ip: req.ip
      });
      res.status(503).json({
        error: {
          message: 'Database unavailable - service cannot authenticate requests',
          statusCode: 503
        }
      });
      return;
    }
  } catch (error: any) {
    logger.error('Extension authentication failed', { error: error.message });
    res.status(401).json({
      error: {
        message: 'Invalid API key',
        statusCode: 401
      }
    });
    return;
  }
};
