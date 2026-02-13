import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';

/**
 * Latency tracking statistics.
 * Tracks p50, p95, p99 per endpoint path.
 */
interface LatencyStats {
  count: number;
  latencies: number[];
  p50: number;
  p95: number;
  p99: number;
  mean: number;
}

const endpointStats: Record<string, LatencyStats> = {};
const MAX_SAMPLES = 1000; // Rolling window

function getPercentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const index = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, index)];
}

function updateStats(path: string, latencyMs: number): void {
  if (!endpointStats[path]) {
    endpointStats[path] = {
      count: 0,
      latencies: [],
      p50: 0,
      p95: 0,
      p99: 0,
      mean: 0,
    };
  }

  const stats = endpointStats[path];
  stats.count++;
  stats.latencies.push(latencyMs);

  // Keep rolling window
  if (stats.latencies.length > MAX_SAMPLES) {
    stats.latencies.shift();
  }

  // Recalculate percentiles
  const sorted = [...stats.latencies].sort((a, b) => a - b);
  stats.p50 = getPercentile(sorted, 50);
  stats.p95 = getPercentile(sorted, 95);
  stats.p99 = getPercentile(sorted, 99);
  stats.mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
}

/**
 * Middleware that tracks request latency and adds timing headers.
 */
export function latencyMiddleware(req: Request, res: Response, next: NextFunction): void {
  const start = process.hrtime.bigint();

  res.on('finish', () => {
    const end = process.hrtime.bigint();
    const durationMs = Number(end - start) / 1_000_000;

    // Normalize path (remove IDs)
    const normalizedPath = req.path.replace(/[0-9a-f-]{36}/g, ':id');

    updateStats(normalizedPath, durationMs);

    // Add timing header
    res.setHeader('X-Response-Time', `${durationMs.toFixed(2)}ms`);

    // Log slow requests
    if (durationMs > 100) {
      logger.warn('Slow request detected', {
        path: normalizedPath,
        method: req.method,
        duration_ms: Math.round(durationMs),
        status: res.statusCode,
      });
    }
  });

  next();
}

/**
 * Get latency statistics for all endpoints.
 */
export function getLatencyStats(): Record<string, Omit<LatencyStats, 'latencies'>> {
  const result: Record<string, Omit<LatencyStats, 'latencies'>> = {};
  for (const [path, stats] of Object.entries(endpointStats)) {
    result[path] = {
      count: stats.count,
      p50: Math.round(stats.p50 * 100) / 100,
      p95: Math.round(stats.p95 * 100) / 100,
      p99: Math.round(stats.p99 * 100) / 100,
      mean: Math.round(stats.mean * 100) / 100,
    };
  }
  return result;
}
