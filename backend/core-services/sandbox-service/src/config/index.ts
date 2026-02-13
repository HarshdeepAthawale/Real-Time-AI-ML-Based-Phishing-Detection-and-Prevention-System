import dotenv from 'dotenv';

dotenv.config();

export interface SandboxConfig {
  provider: 'cuckoo' | 'anyrun' | 'disabled';
  cuckoo: {
    url?: string;
    apiKey?: string;
  };
  anyrun: {
    apiKey?: string;
  };
  timeout: number;
  pollInterval: number;
  queue: {
    concurrency: number;
  };
  cors: {
    origin: string | string[];
    credentials: boolean;
  };
}

const getConfig = (): SandboxConfig => {
  const corsOrigins = process.env.CORS_ORIGINS;
  let corsOrigin: string | string[] = '*';
  
  if (corsOrigins) {
    if (corsOrigins.includes(',')) {
      corsOrigin = corsOrigins.split(',').map(origin => origin.trim());
    } else if (corsOrigins !== '*') {
      corsOrigin = corsOrigins;
    }
  }

  let provider = (process.env.SANDBOX_PROVIDER || 'anyrun') as 'cuckoo' | 'anyrun' | 'disabled';
  
  // Auto-disable if required keys are missing
  if (provider === 'cuckoo' && !process.env.CUCKOO_SANDBOX_URL) {
    console.warn('CUCKOO_SANDBOX_URL not set; sandbox analysis disabled. Set SANDBOX_PROVIDER=disabled to silence this warning.');
    provider = 'disabled';
  }
  
  if (provider === 'anyrun' && !process.env.ANYRUN_API_KEY) {
    console.warn('ANYRUN_API_KEY not set; sandbox analysis disabled. Set SANDBOX_PROVIDER=disabled to silence this warning.');
    provider = 'disabled';
  }

  return {
    provider,
    cuckoo: {
      url: process.env.CUCKOO_SANDBOX_URL,
      apiKey: process.env.CUCKOO_API_KEY,
    },
    anyrun: {
      apiKey: process.env.ANYRUN_API_KEY,
    },
    timeout: parseInt(process.env.SANDBOX_TIMEOUT || '300', 10),
    pollInterval: parseInt(process.env.SANDBOX_POLL_INTERVAL || '10', 10),
    queue: {
      concurrency: parseInt(process.env.SANDBOX_QUEUE_CONCURRENCY || '5', 10),
    },
    cors: {
      origin: corsOrigin,
      credentials: true,
    },
  };
};

export const config = getConfig();
