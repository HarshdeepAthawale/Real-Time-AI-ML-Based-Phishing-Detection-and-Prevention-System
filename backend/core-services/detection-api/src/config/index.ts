import dotenv from 'dotenv';

dotenv.config();

export const config = {
  port: parseInt(process.env.PORT || '3001', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  
  mlServices: {
    nlp: process.env.NLP_SERVICE_URL || 'http://nlp-service:8000',
    url: process.env.URL_SERVICE_URL || 'http://url-service:8000',
    visual: process.env.VISUAL_SERVICE_URL || 'http://visual-service:8000'
  },
  
  redis: {
    host: process.env.REDIS_HOST || 'redis',
    port: parseInt(process.env.REDIS_PORT || '6379', 10),
    password: process.env.REDIS_PASSWORD || undefined,
    url: process.env.REDIS_URL || 'redis://redis:6379'
  },
  
  cors: {
    origin: process.env.CORS_ORIGINS 
      ? (process.env.CORS_ORIGINS.includes(',') 
          ? process.env.CORS_ORIGINS.split(',') 
          : process.env.CORS_ORIGINS === '*' 
            ? '*' 
            : [process.env.CORS_ORIGINS])
      : '*',
    credentials: true
  },
  
  rateLimit: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // requests per window
  },
  
  cache: {
    defaultTtl: 3600, // 1 hour
    urlTtl: 7200 // 2 hours for URLs
  }
};
