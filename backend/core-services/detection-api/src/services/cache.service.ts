import Redis from 'ioredis';
import { config } from '../config';
import { logger } from '../utils/logger';
import crypto from 'crypto';

export class CacheService {
  private client: Redis | null = null;
  private connectionAttempted = false;
  
  constructor() {
    try {
      this.client = new Redis({
        host: config.redis.host,
        port: config.redis.port,
        password: config.redis.password,
        retryStrategy: (times) => {
          const delay = Math.min(times * 50, 2000);
          return delay;
        },
        enableOfflineQueue: false, // Don't queue commands when offline
        maxRetriesPerRequest: 1,
        lazyConnect: true, // Don't connect immediately
        connectTimeout: 5000,
        enableReadyCheck: true
      });
      
      this.client.on('error', (err) => {
        logger.debug('Redis connection error (cache operations will be skipped):', err.message);
      });
      
      this.client.on('connect', () => {
        logger.info('Redis cache connected successfully');
      });
      
      this.client.on('ready', () => {
        logger.info('Redis cache ready');
      });
      
      this.client.on('close', () => {
        logger.debug('Redis connection closed');
      });
      
      // Attempt to connect asynchronously (non-blocking)
      this.attemptConnection();
    } catch (error: any) {
      logger.warn('Failed to initialize Redis client (cache disabled):', error.message);
      this.client = null;
    }
  }
  
  private async attemptConnection(): Promise<void> {
    if (this.connectionAttempted || !this.client) {
      return;
    }
    
    this.connectionAttempted = true;
    try {
      await this.client.connect();
      logger.info('Redis cache connection established');
    } catch (err: any) {
      logger.warn(`Redis cache unavailable (service will continue without cache): ${err.message}`);
      // Service continues without cache - this is acceptable
    }
  }
  
  async get(key: string): Promise<any | null> {
    if (!this.client || !this.client.status || this.client.status !== 'ready') {
      return null; // Cache unavailable, return null gracefully
    }
    
    try {
      const value = await this.client.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error: any) {
      // Log only if it's not a connection error (those are expected)
      if (error.message && !error.message.includes('Connection')) {
        logger.debug('Cache get error (non-critical):', error.message);
      }
      return null;
    }
  }
  
  async set(key: string, value: any, ttlSeconds: number = 3600): Promise<void> {
    if (!this.client || !this.client.status || this.client.status !== 'ready') {
      return; // Cache unavailable, skip silently
    }
    
    try {
      await this.client.setex(key, ttlSeconds, JSON.stringify(value));
    } catch (error: any) {
      // Log only if it's not a connection error
      if (error.message && !error.message.includes('Connection')) {
        logger.debug('Cache set error (non-critical):', error.message);
      }
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
    if (!this.client) {
      return false;
    }
    
    try {
      if (this.client.status === 'ready') {
        await this.client.ping();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }
  
  async disconnect(): Promise<void> {
    if (this.client) {
      try {
        await this.client.quit();
        logger.info('Redis cache disconnected');
      } catch (error: any) {
        logger.debug('Error disconnecting Redis:', error.message);
      }
    }
  }
}
