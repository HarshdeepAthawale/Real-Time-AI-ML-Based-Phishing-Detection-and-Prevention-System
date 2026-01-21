import Redis from 'ioredis';

// Mock logger for tests to avoid console output
const logger = {
  info: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn(),
};

let testRedis: Redis | null = null;
let redisAvailable: boolean | null = null;

// Reset cache function for testing
export function resetRedisAvailabilityCache(): void {
  redisAvailable = null;
}

const TEST_KEY_PREFIX = 'test:threat-intel:';

/**
 * Check if Redis is available
 */
export async function checkRedisAvailability(): Promise<boolean> {
  if (redisAvailable !== null) {
    return redisAvailable;
  }

  const testRedisUrl = process.env.TEST_REDIS_URL || 
    process.env.REDIS_URL ||
    'redis://localhost:6379/1';

  const tempRedis = new Redis(testRedisUrl, {
    maxRetriesPerRequest: 1,
    retryStrategy: () => null, // Don't retry for availability check
    connectTimeout: 2000,
    lazyConnect: true,
  });

  try {
    await tempRedis.connect();
    await tempRedis.ping();
    await tempRedis.quit();
    redisAvailable = true;
    return true;
  } catch (error) {
    redisAvailable = false;
    return false;
  }
}

/**
 * Setup test Redis connection
 */
export async function setupTestRedis(): Promise<Redis> {
  if (testRedis?.status === 'ready') {
    return testRedis;
  }

  // Check availability first
  const isAvailable = await checkRedisAvailability();
  if (!isAvailable) {
    const testRedisUrl = process.env.TEST_REDIS_URL || 
      process.env.REDIS_URL ||
      'redis://localhost:6379/1';
    
    throw new Error(
      `Test Redis is not available at ${testRedisUrl}. ` +
      `\n\nTo run integration tests:` +
      `\n1. Make sure Redis is running: redis-server` +
      `\n2. Set TEST_REDIS_URL if using different connection string` +
      `\n\nIntegration tests will be skipped if Redis is unavailable.`
    );
  }

  const testRedisUrl = process.env.TEST_REDIS_URL || 
    process.env.REDIS_URL ||
    'redis://localhost:6379/1'; // Use database 1 for tests

  testRedis = new Redis(testRedisUrl, {
    maxRetriesPerRequest: 3,
    retryStrategy: (times) => {
      const delay = Math.min(times * 50, 2000);
      return delay;
    },
    keyPrefix: TEST_KEY_PREFIX, // Prefix all keys for test isolation
  });

  testRedis.on('error', (err: any) => {
    const errorMessage = err?.message || String(err);
    if (errorMessage.includes('ECONNREFUSED') || errorMessage.includes('connect')) {
      logger.error(
        `Cannot connect to test Redis at ${testRedisUrl}. ` +
        `Make sure Redis is running and accessible. ` +
        `Set TEST_REDIS_URL environment variable if using a different Redis instance.`
      );
    } else {
      logger.error('Test Redis connection error', err);
    }
  });

  testRedis.on('connect', () => {
    logger.info('Test Redis connected');
  });

  // Wait for connection to be ready
  if (testRedis.status !== 'ready') {
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Redis connection timeout'));
      }, 5000);

      testRedis!.on('ready', () => {
        clearTimeout(timeout);
        resolve();
      });

      testRedis!.on('error', (err) => {
        clearTimeout(timeout);
        reject(err);
      });
    });
  }

  return testRedis;
}

/**
 * Get test Redis connection
 */
export function getTestRedis(): Redis {
  if (!testRedis || testRedis.status !== 'ready') {
    throw new Error('Test Redis not connected. Call setupTestRedis() first.');
  }
  return testRedis;
}

/**
 * Clean test Redis - remove all keys with test prefix
 */
export async function cleanupTestRedis(): Promise<void> {
  if (!testRedis || testRedis.status !== 'ready') {
    return;
  }

  try {
    // Get all keys with test prefix
    const keys = await testRedis.keys('*');
    if (keys.length > 0) {
      // Remove key prefix for deletion
      const keysWithoutPrefix = keys.map(key => key.replace(TEST_KEY_PREFIX, ''));
      await testRedis.del(...keysWithoutPrefix);
    }
    logger.debug(`Test Redis cleaned: ${keys.length} keys removed`);
  } catch (error) {
    logger.error('Failed to clean test Redis', error);
    throw error;
  }
}

/**
 * Disconnect from test Redis
 */
export async function disconnectTestRedis(): Promise<void> {
  if (testRedis) {
    await testRedis.quit();
    testRedis = null;
    logger.info('Test Redis disconnected');
  }
}
