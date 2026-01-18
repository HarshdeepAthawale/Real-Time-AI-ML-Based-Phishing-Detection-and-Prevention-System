import { Request, Response, NextFunction } from 'express';

/**
 * Handler for threat intelligence API requests
 * Can be used for request/response transformation
 */
export const threatIntelHandler = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  // Add request metadata
  req.headers['x-service'] = 'threat-intel';
  req.headers['x-request-id'] = req.headers['x-request-id'] || 
    `ti-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  
  next();
};
