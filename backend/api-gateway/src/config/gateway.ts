export interface ServiceConfig {
  name: string;
  url: string;
  timeout: number;
  retries: number;
}

export const serviceConfigs: Record<string, ServiceConfig> = {
  'detection-api': {
    name: 'detection-api',
    url: process.env.DETECTION_API_URL || 'http://detection-api:3001',
    timeout: 30000,
    retries: 3
  },
  'threat-intel': {
    name: 'threat-intel',
    url: process.env.THREAT_INTEL_URL || 'http://threat-intel:3002',
    timeout: 30000,
    retries: 3
  },
  'sandbox-service': {
    name: 'sandbox-service',
    url: process.env.SANDBOX_SERVICE_URL || 'http://sandbox-service:3004',
    timeout: 60000,
    retries: 2
  },
  'extension-api': {
    name: 'extension-api',
    url: process.env.EXTENSION_API_URL || 'http://extension-api:3003',
    timeout: 30000,
    retries: 3
  },
  'nlp-service': {
    name: 'nlp-service',
    url: process.env.NLP_SERVICE_URL || 'http://nlp-service:8000',
    timeout: 60000,
    retries: 2
  },
  'url-service': {
    name: 'url-service',
    url: process.env.URL_SERVICE_URL || 'http://url-service:8001',
    timeout: 30000,
    retries: 2
  },
  'visual-service': {
    name: 'visual-service',
    url: process.env.VISUAL_SERVICE_URL || 'http://visual-service:8002',
    timeout: 60000,
    retries: 2
  }
};

export const routeConfig = {
  '/api/v1/detect': 'detection-api',
  '/api/v1/intelligence': 'threat-intel',
  '/api/v1/dashboard': 'detection-api',
  '/api/v1/ioc': 'threat-intel',
  '/api/v1/feeds': 'threat-intel',
  '/api/v1/sync': 'threat-intel',
  '/api/v1/sandbox': 'sandbox-service',
  '/api/v1/extension': 'extension-api',
  '/api/v1/nlp': 'nlp-service',
  '/api/v1/url': 'url-service',
  '/api/v1/visual': 'visual-service'
  // WebSocket route handled separately in routes/index.ts
};

export const rateLimitConfig = {
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each API key to 100 requests per windowMs
  standardHeaders: true,
  legacyHeaders: false
};
