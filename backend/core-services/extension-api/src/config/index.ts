import dotenv from 'dotenv';

dotenv.config();

export const config = {
  port: parseInt(process.env.EXTENSION_API_PORT || process.env.PORT || '3003', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  
  detectionApi: {
    url: process.env.DETECTION_API_URL || 'http://detection-api:3001',
    timeout: {
      url: parseInt(process.env.DETECTION_API_TIMEOUT_URL || '5000', 10),
      email: parseInt(process.env.DETECTION_API_TIMEOUT_EMAIL || '10000', 10),
    }
  },
  
  threatIntel: {
    url: process.env.THREAT_INTEL_URL || 'http://threat-intel:3002',
  },
  
  redis: {
    host: process.env.REDIS_HOST || process.env.REDIS_URL?.split('://')[1]?.split(':')[0] || 'redis',
    port: parseInt(process.env.REDIS_PORT || process.env.REDIS_URL?.split(':')[2]?.split('/')[0] || '6379', 10),
    password: process.env.REDIS_PASSWORD || undefined,
    url: process.env.REDIS_URL || 'redis://redis:6379'
  },
  
  cors: {
    origin: process.env.CORS_ORIGINS 
      ? (process.env.CORS_ORIGINS.includes(',') 
          ? process.env.CORS_ORIGINS.split(',').map(o => o.trim())
          : [process.env.CORS_ORIGINS.trim()])
      : [
          'chrome-extension://*',
          'moz-extension://*',
          'ms-browser-extension://*'
        ],
    credentials: true
  },
  
  rateLimit: {
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '900000', 10), // 15 minutes
    max: parseInt(process.env.RATE_LIMIT_MAX || '100', 10), // requests per window
    extensionMax: parseInt(process.env.RATE_LIMIT_EXTENSION_MAX || '200', 10), // higher limit for extensions
  },
  
  cache: {
    defaultTtl: parseInt(process.env.CACHE_TTL_DEFAULT || '1800', 10), // 30 minutes for extension API
    urlTtl: parseInt(process.env.CACHE_TTL_URL || '1800', 10), // 30 minutes
    emailTtl: parseInt(process.env.CACHE_TTL_EMAIL || '1800', 10), // 30 minutes
  },
  
  extension: {
    apiKeySecret: process.env.EXTENSION_API_KEY_SECRET || process.env.JWT_SECRET || 'change-me-in-production',
  }
};
