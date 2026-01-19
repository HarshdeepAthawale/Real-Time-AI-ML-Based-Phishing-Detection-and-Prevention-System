import Redis from 'ioredis';
import { Queue, Worker, QueueEvents } from 'bullmq';
import { config } from '../../config';
import { QUEUE_NAMES } from './queue-keys';

let redisClient: Redis | null = null;
let queues: Map<string, Queue> = new Map();
let workers: Map<string, Worker> = new Map();

export const connectRedis = async (): Promise<Redis> => {
  if (redisClient) {
    return redisClient;
  }

  try {
    redisClient = new Redis(config.redis.url, {
      maxRetriesPerRequest: 3,
      retryStrategy: (times) => {
        const delay = Math.min(times * 50, 2000);
        return delay;
      },
    });

    redisClient.on('error', (err) => {
      console.error('Redis connection error:', err);
    });

    redisClient.on('connect', () => {
      console.log('Connected to Redis');
    });

    return redisClient;
  } catch (error) {
    console.error('Failed to connect to Redis:', error);
    throw error;
  }
};

export const getRedis = (): Redis => {
  if (!redisClient) {
    throw new Error('Redis not connected. Call connectRedis() first.');
  }
  return redisClient;
};

export const disconnectRedis = async (): Promise<void> => {
  // Close all workers
  for (const worker of workers.values()) {
    await worker.close();
  }
  workers.clear();

  // Close all queues
  for (const queue of queues.values()) {
    await queue.close();
  }
  queues.clear();

  // Close Redis connection
  if (redisClient) {
    await redisClient.quit();
    redisClient = null;
  }
};

// Parse Redis URL to get host and port
const parseRedisUrl = (url: string): { host: string; port: number } => {
  try {
    const parsed = new URL(url);
    return {
      host: parsed.hostname || 'localhost',
      port: parseInt(parsed.port || '6379', 10),
    };
  } catch {
    return { host: 'localhost', port: 6379 };
  }
};

// Queue management
export const getQueue = (queueName: string): Queue => {
  if (queues.has(queueName)) {
    return queues.get(queueName)!;
  }

  const connection = parseRedisUrl(config.redis.url);
  const queue = new Queue(queueName, {
    connection,
  });

  queues.set(queueName, queue);
  return queue;
};

export const createWorker = <T = any>(
  queueName: string,
  processor: (job: { data: T }) => Promise<any>
): Worker => {
  if (workers.has(queueName)) {
    return workers.get(queueName)!;
  }

  const connection = parseRedisUrl(config.redis.url);
  const worker = new Worker(queueName, processor, {
    connection,
  });

  workers.set(queueName, worker);
  return worker;
};

export const getQueueEvents = (queueName: string): QueueEvents => {
  const connection = parseRedisUrl(config.redis.url);
  return new QueueEvents(queueName, {
    connection,
  });
};

// Helper to get all queues
export const getAllQueues = (): Map<string, Queue> => {
  return queues;
};
