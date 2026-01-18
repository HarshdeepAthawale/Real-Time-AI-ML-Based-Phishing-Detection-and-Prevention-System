import dotenv from 'dotenv';

dotenv.config();

export interface Config {
  environment: string;
  database: {
    url: string;
    mongodbUrl: string;
  };
  redis: {
    url: string;
  };
  aws: {
    region: string;
    s3: {
      models: string;
      training: string;
      logs: string;
      artifacts: string;
    };
  };
  services: {
    nlp: string;
    url: string;
    visual: string;
    detectionApi: string;
    threatIntel: string;
  };
  security: {
    jwtSecret: string;
    apiKeyEncryptionKey: string;
  };
  monitoring: {
    logLevel: string;
    sentryDsn?: string;
  };
  app: {
    port: number;
    nodeEnv: string;
  };
}

const getConfig = (): Config => {
  const environment = process.env.NODE_ENV || 'development';

  return {
    environment,
    database: {
      url: process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/phishing_detection',
      mongodbUrl: process.env.MONGODB_URL || 'mongodb://localhost:27017/phishing_detection',
    },
    redis: {
      url: process.env.REDIS_URL || 'redis://localhost:6379',
    },
    aws: {
      region: process.env.AWS_REGION || 'us-east-1',
      s3: {
        models: process.env.S3_BUCKET_MODELS || 'phishing-detection-models-dev',
        training: process.env.S3_BUCKET_TRAINING || 'phishing-detection-training-dev',
        logs: process.env.S3_BUCKET_LOGS || 'phishing-detection-logs-dev',
        artifacts: process.env.S3_BUCKET_ARTIFACTS || 'phishing-detection-artifacts-dev',
      },
    },
    services: {
      nlp: process.env.NLP_SERVICE_URL || 'http://localhost:8000',
      url: process.env.URL_SERVICE_URL || 'http://localhost:8000',
      visual: process.env.VISUAL_SERVICE_URL || 'http://localhost:8000',
      detectionApi: process.env.DETECTION_API_URL || 'http://localhost:3001',
      threatIntel: process.env.THREAT_INTEL_URL || 'http://localhost:3002',
    },
    security: {
      jwtSecret: process.env.JWT_SECRET || 'change-me-in-production',
      apiKeyEncryptionKey: process.env.API_KEY_ENCRYPTION_KEY || 'change-me-in-production',
    },
    monitoring: {
      logLevel: process.env.LOG_LEVEL || 'info',
      sentryDsn: process.env.SENTRY_DSN,
    },
    app: {
      port: parseInt(process.env.PORT || '3000', 10),
      nodeEnv: environment,
    },
  };
};

export const config = getConfig();
