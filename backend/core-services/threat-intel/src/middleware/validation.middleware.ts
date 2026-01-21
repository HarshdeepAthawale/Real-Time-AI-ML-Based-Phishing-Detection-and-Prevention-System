import { Request, Response, NextFunction } from 'express';
import { z, ZodSchema } from 'zod';
import { ParamsDictionary } from 'express-serve-static-core';

/**
 * Validate request body with Zod schema
 */
export function validateBody<T>(schema: ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      req.body = schema.parse(req.body) as T;
      next();
    } catch (error) {
      next(error);
    }
  };
}

/**
 * Validate request query parameters with Zod schema
 */
export function validateQuery<T extends Record<string, any>>(schema: ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      req.query = schema.parse(req.query) as T;
      next();
    } catch (error) {
      next(error);
    }
  };
}

/**
 * Validate request params with Zod schema
 */
export function validateParams<T extends ParamsDictionary>(schema: ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      req.params = schema.parse(req.params) as T;
      next();
    } catch (error) {
      next(error);
    }
  };
}
