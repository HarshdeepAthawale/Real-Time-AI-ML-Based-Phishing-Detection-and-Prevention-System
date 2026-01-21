import Redis from 'ioredis';

let testRedisClient: Redis | null = null;

/**
 * Get test Redis URL from environment
 */
export const getTestRedisUrl = (): string => {
  return process.env.TEST_REDIS_URL || 'redis://localhost:6380';
};

/**
 * Connect to test Redis
 */
export const connectTestRedis = async (): Promise<Redis> => {
  if (testRedisClient && testRedisClient.status === 'ready') {
    return testRedisClient;
  }

  const redisUrl = getTestRedisUrl();
  
  testRedisClient = new Redis(redisUrl, {
    maxRetriesPerRequest: 3,
    retryStrategy: (times) => {
      const delay = Math.min(times * 50, 2000);
      return delay;
    },
    enableReadyCheck: true,
    lazyConnect: false,
  });

  testRedisClient.on('error', (err) => {
    console.error('Test Redis connection error:', err);
  });

  try {
    await testRedisClient.connect();
    return testRedisClient;
  } catch (error) {
    console.error('Failed to connect to test Redis:', error);
    throw error;
  }
};

/**
 * Get test Redis client
 */
export const getTestRedis = (): Redis => {
  if (!testRedisClient || testRedisClient.status !== 'ready') {
    throw new Error('Test Redis not connected. Call connectTestRedis() first.');
  }
  return testRedisClient;
};

/**
 * Flush all data from test Redis
 */
export const flushTestRedis = async (): Promise<void> => {
  if (testRedisClient && testRedisClient.status === 'ready') {
    await testRedisClient.flushall();
  }
};

/**
 * Disconnect from test Redis
 */
export const disconnectTestRedis = async (): Promise<void> => {
  if (testRedisClient) {
    await testRedisClient.quit();
    testRedisClient = null;
  }
};

/**
 * Check if test Redis is available
 */
export const isTestRedisAvailable = async (): Promise<boolean> => {
  try {
    const client = await connectTestRedis();
    await client.ping();
    return true;
  } catch {
    return false;
  }
};
