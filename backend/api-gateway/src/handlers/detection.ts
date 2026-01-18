import { Request, Response, NextFunction } from 'express';

/**
 * Handler for detection API requests
 * Can be used for request/response transformation
 */
export const detectionHandler = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  // Add request metadata
  req.headers['x-service'] = 'detection-api';
  req.headers['x-request-id'] = req.headers['x-request-id'] || 
    `det-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  
  next();
};
