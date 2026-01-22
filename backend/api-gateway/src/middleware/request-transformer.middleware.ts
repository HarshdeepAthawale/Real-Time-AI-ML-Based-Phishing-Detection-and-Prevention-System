import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';
import { v4 as uuidv4 } from 'uuid';

/**
 * Request Transformation Middleware
 * 
 * Features:
 * - Add request ID for tracing
 * - Transform request/response format
 * - Add metadata (timestamp, version, etc.)
 * - Request/response logging
 * - NO MOCKS - real transformations
 */

export interface TransformedRequest extends Request {
  requestId: string;
  startTime: number;
  metadata: {
    timestamp: string;
    version: string;
    source: string;
    ip: string;
  };
}

/**
 * Add request ID and metadata
 */
export function requestEnhancementMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const transformedReq = req as TransformedRequest;
  
  // Generate unique request ID
  transformedReq.requestId = uuidv4();
  transformedReq.startTime = Date.now();
  
  // Add metadata
  transformedReq.metadata = {
    timestamp: new Date().toISOString(),
    version: 'v1',
    source: req.get('User-Agent') || 'unknown',
    ip: req.ip || req.socket.remoteAddress || 'unknown'
  };

  // Add request ID to response headers
  res.setHeader('X-Request-ID', transformedReq.requestId);
  res.setHeader('X-API-Version', 'v1');

  // Log request
  logger.info('Incoming request', {
    requestId: transformedReq.requestId,
    method: req.method,
    path: req.path,
    ip: transformedReq.metadata.ip
  });

  next();
}

/**
 * Transform request body to standardized format
 */
export function requestBodyTransformerMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  if (req.body && Object.keys(req.body).length > 0) {
    // Add request metadata to body
    req.body._metadata = {
      request_id: (req as TransformedRequest).requestId,
      timestamp: new Date().toISOString(),
      source: 'api-gateway'
    };
  }

  next();
}

/**
 * Transform response to standardized format
 */
export function responseTransformerMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const originalJson = res.json.bind(res);
  const transformedReq = req as TransformedRequest;

  res.json = function(data: any): Response {
    // Wrap response in standard envelope
    const transformedResponse = {
      success: res.statusCode < 400,
      data: res.statusCode < 400 ? data : undefined,
      error: res.statusCode >= 400 ? data : undefined,
      metadata: {
        request_id: transformedReq.requestId,
        timestamp: new Date().toISOString(),
        processing_time_ms: Date.now() - transformedReq.startTime,
        version: 'v1'
      }
    };

    return originalJson(transformedResponse);
  };

  next();
}

/**
 * Log response
 */
export function responseLoggingMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const transformedReq = req as TransformedRequest;

  // Capture response
  const originalSend = res.send.bind(res);
  
  res.send = function(data: any): Response {
    const processingTime = Date.now() - transformedReq.startTime;

    logger.info('Outgoing response', {
      requestId: transformedReq.requestId,
      method: req.method,
      path: req.path,
      statusCode: res.statusCode,
      processingTime
    });

    return originalSend(data);
  };

  next();
}

/**
 * Error transformation middleware
 */
export function errorTransformerMiddleware(
  err: any,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const transformedReq = req as TransformedRequest;

  logger.error('Request error', {
    requestId: transformedReq.requestId,
    error: err.message,
    stack: err.stack
  });

  const errorResponse = {
    success: false,
    error: {
      message: err.message || 'Internal server error',
      code: err.code || 'INTERNAL_ERROR',
      details: process.env.NODE_ENV === 'development' ? err.stack : undefined
    },
    metadata: {
      request_id: transformedReq.requestId,
      timestamp: new Date().toISOString(),
      processing_time_ms: Date.now() - transformedReq.startTime
    }
  };

  res.status(err.statusCode || 500).json(errorResponse);
}

/**
 * Request validation middleware
 */
export function requestValidationMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Validate content type for POST/PUT requests
  if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
    const contentType = req.get('Content-Type');
    
    if (!contentType || !contentType.includes('application/json')) {
      return res.status(415).json({
        success: false,
        error: {
          message: 'Unsupported Media Type',
          code: 'INVALID_CONTENT_TYPE',
          details: 'Content-Type must be application/json'
        }
      });
    }
  }

  // Validate body size
  const contentLength = parseInt(req.get('Content-Length') || '0');
  const maxSize = 10 * 1024 * 1024; // 10MB

  if (contentLength > maxSize) {
    return res.status(413).json({
      success: false,
      error: {
        message: 'Payload Too Large',
        code: 'PAYLOAD_TOO_LARGE',
        details: `Maximum payload size is ${maxSize} bytes`
      }
    });
  }

  next();
}
