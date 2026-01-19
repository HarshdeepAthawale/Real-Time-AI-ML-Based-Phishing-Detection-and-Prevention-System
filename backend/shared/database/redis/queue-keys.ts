/**
 * Redis Queue Keys
 * 
 * Job queues (using BullMQ)
 */

// Queue names
export const QUEUE_NAMES = {
  DETECTION_JOBS: 'detection-jobs',
  SANDBOX_JOBS: 'sandbox-jobs',
  TRAINING_JOBS: 'training-jobs',
  THREAT_INTEL_SYNC: 'threat-intel-sync',
};

/**
 * Rate limiting
 * Key: `rate_limit:{api_key}:{window}` (e.g., `rate_limit:abc123:minute`)
 * Value: Counter
 * TTL: Window duration
 */
export const getRateLimitKey = (apiKey: string, window: string): string => {
  return `rate_limit:${apiKey}:${window}`;
};

// Rate limit windows
export const RATE_LIMIT_WINDOWS = {
  MINUTE: 'minute',
  HOUR: 'hour',
  DAY: 'day',
};
