import Redis from 'ioredis';
import { config } from '../config';
import { logger } from '../utils/logger';
import crypto from 'crypto';

export class CacheService {
  private client: Redis;
  
  constructor() {
    this.client = new Redis({
      host: config.redis.host,
      port: config.redis.port,
      password: config.redis.password,
      retryStrategy: (times) => {
        const delay = Math.min(times * 50, 2000);
        return delay;
      }
    });
    
    this.client.on('error', (err) => {
      logger.error('Redis error', err);
    });
    
    this.client.on('connect', () => {
      logger.info('Redis connected');
    });
  }
  
  async get(key: string): Promise<any | null> {
    try {
      const value = await this.client.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      logger.error('Cache get error', error);
      return null;
    }
  }
  
  async set(key: string, value: any, ttlSeconds: number = 3600): Promise<void> {
    try {
      await this.client.setex(key, ttlSeconds, JSON.stringify(value));
    } catch (error) {
      logger.error('Cache set error', error);
    }
  }
  
  async getOrSet<T>(
    key: string,
    fetcher: () => Promise<T>,
    ttlSeconds: number = 3600
  ): Promise<T> {
    const cached = await this.get(key);
    if (cached !== null) {
      return cached as T;
    }
    
    const value = await fetcher();
    await this.set(key, value, ttlSeconds);
    return value;
  }
  
  generateCacheKey(type: string, input: string): string {
    const hash = crypto.createHash('sha256').update(input).digest('hex');
    return `detection:${type}:${hash}`;
  }
  
  async isConnected(): Promise<boolean> {
    try {
      await this.client.ping();
      return true;
    } catch {
      return false;
    }
  }
  
  async disconnect(): Promise<void> {
    await this.client.quit();
  }
}
