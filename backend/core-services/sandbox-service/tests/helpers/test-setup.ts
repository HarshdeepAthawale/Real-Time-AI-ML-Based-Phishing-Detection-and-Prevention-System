// Mock environment variables for testing
process.env.NODE_ENV = 'test';
process.env.PORT = '3004';
process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/test_db';
process.env.REDIS_URL = 'redis://localhost:6379';
process.env.SANDBOX_PROVIDER = 'anyrun';
process.env.ANYRUN_API_KEY = 'test-api-key';
process.env.SANDBOX_TIMEOUT = '300';
process.env.SANDBOX_POLL_INTERVAL = '10';

// Mock logger to avoid console output during tests
jest.mock('../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

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
