import express from 'express';
import { errorHandler } from '../../../src/middleware/error-handler.middleware';

/**
 * Create a test Express app with standard middleware
 * 
 * This helper ensures consistent test app setup across all integration tests.
 * Remember to add error handler AFTER routes in your test beforeEach.
 */
export function createTestApp(): express.Application {
  const app = express();
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));
  
  return app;
}

/**
 * Setup error handler middleware on Express app
 * 
 * IMPORTANT: This MUST be called AFTER all routes are registered.
 * The error handler should be the last middleware in the chain.
 */
export function setupErrorHandler(app: express.Application): void {
  app.use(errorHandler);
}
