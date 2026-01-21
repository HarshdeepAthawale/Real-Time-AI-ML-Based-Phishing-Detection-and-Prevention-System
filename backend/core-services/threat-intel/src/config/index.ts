import dotenv from 'dotenv';

dotenv.config();

export interface Config {
  port: number;
  nodeEnv: string;
  database: {
    url: string;
  };
  redis: {
    url: string;
    host?: string;
    port?: number;
    password?: string;
  };
  misp: {
    url?: string;
    apiKey?: string;
  };
  otx: {
    apiKey?: string;
  };
  bloomFilter: {
    defaultSize: number;
    defaultFalsePositiveRate: number;
    redisTtl: number;
  };
  sync: {
    defaultIntervalMinutes: number;
    cronExpression?: string;
  };
  cors: {
    origin: string | string[];
    credentials: boolean;
  };
}

const getConfig = (): Config => {
  const corsOrigins = process.env.CORS_ORIGINS;
  let corsOrigin: string | string[] = '*';
  
  if (corsOrigins) {
    if (corsOrigins.includes(',')) {
      corsOrigin = corsOrigins.split(',').map(origin => origin.trim());
    } else if (corsOrigins !== '*') {
      corsOrigin = corsOrigins;
    }
  }

  return {
    port: parseInt(process.env.PORT || '3002', 10),
    nodeEnv: process.env.NODE_ENV || 'development',
    
    database: {
      url: process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/phishing_detection',
    },
    
    redis: {
      url: process.env.REDIS_URL || 'redis://localhost:6379',
      host: process.env.REDIS_HOST,
      port: process.env.REDIS_PORT ? parseInt(process.env.REDIS_PORT, 10) : undefined,
      password: process.env.REDIS_PASSWORD,
    },
    
    misp: {
      url: process.env.MISP_URL,
      apiKey: process.env.MISP_API_KEY,
    },
    
    otx: {
      apiKey: process.env.OTX_API_KEY,
    },
    
    bloomFilter: {
      defaultSize: parseInt(process.env.BLOOM_FILTER_SIZE || '1000000', 10),
      defaultFalsePositiveRate: parseFloat(process.env.BLOOM_FILTER_FALSE_POSITIVE_RATE || '0.01'),
      redisTtl: parseInt(process.env.BLOOM_FILTER_TTL || '604800', 10), // 7 days default
    },
    
    sync: {
      defaultIntervalMinutes: parseInt(process.env.SYNC_INTERVAL_MINUTES || '60', 10),
      cronExpression: process.env.SYNC_CRON_EXPRESSION,
    },
    
    cors: {
      origin: corsOrigin,
      credentials: true,
    },
  };
};

export const config = getConfig();
