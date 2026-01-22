import { Request, Response, NextFunction } from 'express';
import { Redis } from 'ioredis';
import { logger } from '../utils/logger';

/**
 * Metrics Collection Middleware
 * 
 * Tracks:
 * - Request count per endpoint
 * - Response times
 * - Error rates
 * - Status code distribution
 * - NO MOCKS - real metrics in Redis
 */

export class MetricsCollector {
  private redis: Redis;

  constructor() {
    this.redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
    logger.info('Metrics collector initialized');
  }

  /**
   * Metrics middleware
   */
  middleware() {
    return async (req: Request, res: Response, next: NextFunction) => {
      const startTime = Date.now();
      const endpoint = this.normalizeEndpoint(req.path);

      // Capture response
      const originalSend = res.send.bind(res);
      
      res.send = (data: any): Response => {
        const responseTime = Date.now() - startTime;
        
        // Record metrics asynchronously (don't block response)
        this.recordMetrics({
          endpoint,
          method: req.method,
          statusCode: res.statusCode,
          responseTime,
          timestamp: Date.now()
        }).catch(err => {
          logger.error(`Failed to record metrics: ${err.message}`);
        });

        return originalSend(data);
      };

      next();
    };
  }

  /**
   * Record metrics to Redis
   */
  private async recordMetrics(data: {
    endpoint: string;
    method: string;
    statusCode: number;
    responseTime: number;
    timestamp: number;
  }): Promise<void> {
    const { endpoint, method, statusCode, responseTime, timestamp } = data;
    const key = `metrics:${endpoint}:${method}`;
    const date = new Date(timestamp).toISOString().split('T')[0];

    // Increment request counter
    await this.redis.hincrby(`${key}:count:${date}`, 'total', 1);

    // Increment status code counter
    await this.redis.hincrby(`${key}:status:${date}`, String(statusCode), 1);

    // Record response time
    await this.redis.zadd(`${key}:times:${date}`, responseTime, `${timestamp}`);

    // Record error if applicable
    if (statusCode >= 400) {
      await this.redis.hincrby(`${key}:errors:${date}`, String(statusCode), 1);
    }

    // Set expiry (keep for 30 days)
    await this.redis.expire(`${key}:count:${date}`, 30 * 86400);
    await this.redis.expire(`${key}:status:${date}`, 30 * 86400);
    await this.redis.expire(`${key}:times:${date}`, 30 * 86400);
    await this.redis.expire(`${key}:errors:${date}`, 30 * 86400);
  }

  /**
   * Get metrics for endpoint
   */
  async getEndpointMetrics(endpoint: string, method: string, date?: string): Promise<{
    total_requests: number;
    status_codes: { [code: string]: number };
    avg_response_time: number;
    p95_response_time: number;
    error_rate: number;
  }> {
    const targetDate = date || new Date().toISOString().split('T')[0];
    const key = `metrics:${endpoint}:${method}`;

    // Get request count
    const countData = await this.redis.hgetall(`${key}:count:${targetDate}`);
    const totalRequests = parseInt(countData.total || '0');

    // Get status codes
    const statusData = await this.redis.hgetall(`${key}:status:${targetDate}`);
    const statusCodes: { [code: string]: number } = {};
    Object.entries(statusData).forEach(([code, count]) => {
      statusCodes[code] = parseInt(count as string);
    });

    // Get response times
    const times = await this.redis.zrange(`${key}:times:${targetDate}`, 0, -1, 'WITHSCORES');
    const responseTimes: number[] = [];
    for (let i = 1; i < times.length; i += 2) {
      responseTimes.push(parseInt(times[i]));
    }

    const avgResponseTime = responseTimes.length > 0
      ? responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length
      : 0;

    const sortedTimes = responseTimes.sort((a, b) => a - b);
    const p95ResponseTime = sortedTimes.length > 0
      ? sortedTimes[Math.floor(sortedTimes.length * 0.95)]
      : 0;

    // Calculate error rate
    const errorData = await this.redis.hgetall(`${key}:errors:${targetDate}`);
    const totalErrors = Object.values(errorData).reduce(
      (sum, count) => sum + parseInt(count as string), 
      0
    );
    const errorRate = totalRequests > 0 ? (totalErrors / totalRequests) * 100 : 0;

    return {
      total_requests: totalRequests,
      status_codes: statusCodes,
      avg_response_time: Math.round(avgResponseTime),
      p95_response_time: Math.round(p95ResponseTime),
      error_rate: Math.round(errorRate * 100) / 100
    };
  }

  /**
   * Get aggregated metrics
   */
  async getAggregatedMetrics(date?: string): Promise<{
    total_requests: number;
    endpoints: Array<{
      endpoint: string;
      method: string;
      requests: number;
      avg_response_time: number;
      error_rate: number;
    }>;
  }> {
    const targetDate = date || new Date().toISOString().split('T')[0];
    const pattern = `metrics:*:count:${targetDate}`;
    const keys = await this.redis.keys(pattern);

    const endpoints = [];
    let totalRequests = 0;

    for (const key of keys) {
      const parts = key.split(':');
      const endpoint = parts[1];
      const method = parts[2];

      const metrics = await this.getEndpointMetrics(endpoint, method, targetDate);
      totalRequests += metrics.total_requests;

      endpoints.push({
        endpoint,
        method,
        requests: metrics.total_requests,
        avg_response_time: metrics.avg_response_time,
        error_rate: metrics.error_rate
      });
    }

    return {
      total_requests: totalRequests,
      endpoints: endpoints.sort((a, b) => b.requests - a.requests)
    };
  }

  /**
   * Normalize endpoint (remove IDs, etc.)
   */
  private normalizeEndpoint(path: string): string {
    // Replace UUIDs with :id
    let normalized = path.replace(
      /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/gi,
      ':id'
    );

    // Replace numeric IDs
    normalized = normalized.replace(/\/\d+\//g, '/:id/');
    normalized = normalized.replace(/\/\d+$/g, '/:id');

    return normalized;
  }

  /**
   * Close Redis connection
   */
  async close(): Promise<void> {
    await this.redis.quit();
  }
}

/**
 * Create metrics collector instance
 */
export const metricsCollector = new MetricsCollector();

/**
 * Export middleware
 */
export const metricsMiddleware = metricsCollector.middleware();
