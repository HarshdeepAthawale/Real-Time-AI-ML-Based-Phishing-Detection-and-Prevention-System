import { Express } from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import express from 'express';
import { config } from '../../src/config';

// Mock environment variables for testing
process.env.NODE_ENV = 'test';
process.env.PORT = '3001';
process.env.REDIS_HOST = 'localhost';
process.env.REDIS_PORT = '6379';
process.env.NLP_SERVICE_URL = 'http://localhost:8001';
process.env.URL_SERVICE_URL = 'http://localhost:8002';
process.env.VISUAL_SERVICE_URL = 'http://localhost:8003';

// Mock logger to avoid console output during tests
jest.mock('../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

export const createTestApp = (): Express => {
  const app = express();
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));
  return app;
};

export const createTestServer = (app: Express) => {
  const httpServer = createServer(app);
  const io = new Server(httpServer, {
    cors: {
      origin: '*',
      methods: ['GET', 'POST'],
    },
  });
  return { httpServer, io };
};

// Global test setup
beforeAll(() => {
  // Setup before all tests
});

afterAll(() => {
  // Cleanup after all tests
});

// Clean up after each test
afterEach(() => {
  jest.clearAllMocks();
});
