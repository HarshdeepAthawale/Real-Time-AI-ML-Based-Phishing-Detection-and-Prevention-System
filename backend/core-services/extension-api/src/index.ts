import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { config } from './config';
import { logger } from './utils/logger';
import { CacheService } from './services/cache.service';
import { PrivacyFilterService } from './services/privacy-filter.service';
import { URLCheckerService } from './services/url-checker.service';
import { EmailScannerService } from './services/email-scanner.service';
import { extensionAuthMiddleware } from './middleware/extension-auth.middleware';
import { createRateLimitMiddleware } from './middleware/rate-limit.middleware';
import urlCheckRoutes from './routes/url-check.routes';
import emailScanRoutes from './routes/email-scan.routes';
import reportRoutes from './routes/report.routes';
import emailClientRoutes from './routes/email-client.routes';
import { EmailClientService } from './services/email-client.service';
import { connectPostgreSQL, disconnectPostgreSQL, getPostgreSQL } from '../../shared/database';

const app = express();

// Initialize services
const cacheService = new CacheService();
const privacyFilter = new PrivacyFilterService();
const urlChecker = new URLCheckerService(
  config.detectionApi.url,
  cacheService,
  privacyFilter
);
const emailScanner = new EmailScannerService(
  config.detectionApi.url,
  privacyFilter,
  urlChecker
);
const emailClient = new EmailClientService();

// Store services in app context for route access
app.set('cacheService', cacheService);
app.set('privacyFilter', privacyFilter);
app.set('urlChecker', urlChecker);
app.set('emailScanner', emailScanner);
app.set('emailClient', emailClient);

// Middleware
app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' } // Allow extension origins
}));

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps, Postman, or curl)
    if (!origin) {
      return callback(null, true);
    }
    
    // Check if origin matches any allowed pattern
    const allowedOrigins = config.cors.origin;
    
    if (Array.isArray(allowedOrigins)) {
      // Check if origin matches any pattern (supports wildcards)
      const isAllowed = allowedOrigins.some(pattern => {
        if (pattern.includes('*')) {
          const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '$');
          return regex.test(origin);
        }
        return origin === pattern;
      });
      
      if (isAllowed) {
        return callback(null, true);
      }
    } else if (allowedOrigins === '*') {
      return callback(null, true);
    }
    
    callback(new Error('Not allowed by CORS'));
  },
  credentials: config.cors.credentials
}));

app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

// Create rate limit middleware
const rateLimitMiddleware = createRateLimitMiddleware(cacheService);

// Health check endpoint (no auth required)
app.get('/health', async (req, res) => {
  try {
    const cacheConnected = await cacheService.isConnected();
    
    // Check database connection
    let dbConnected = false;
    try {
      const dataSource = getPostgreSQL();
      dbConnected = dataSource.isInitialized;
    } catch {
      dbConnected = false;
    }
    
    res.json({
      status: 'healthy',
      service: 'extension-api',
      cache: cacheConnected ? 'connected' : 'disconnected',
      database: dbConnected ? 'connected' : 'disconnected',
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    logger.error('Health check error', error);
    res.status(503).json({
      status: 'unhealthy',
      service: 'extension-api',
      error: error.message
    });
  }
});

// API routes with authentication and rate limiting
app.use(
  '/api/v1/extension/check-url',
  extensionAuthMiddleware,
  rateLimitMiddleware,
  urlCheckRoutes
);

app.use(
  '/api/v1/extension/scan-email',
  extensionAuthMiddleware,
  rateLimitMiddleware,
  emailScanRoutes
);

app.use(
  '/api/v1/extension/report',
  extensionAuthMiddleware,
  rateLimitMiddleware,
  reportRoutes
);

app.use(
  '/api/v1/extension/email',
  extensionAuthMiddleware,
  rateLimitMiddleware,
  emailClientRoutes
);

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    service: 'Extension API Service',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      checkUrl: '/api/v1/extension/check-url',
      scanEmail: '/api/v1/extension/scan-email',
      report: '/api/v1/extension/report',
      email: '/api/v1/extension/email'
    }
  });
});

// Error handling middleware
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error', {
    error: err.message,
    stack: err.stack,
    path: req.path
  });
  
  res.status(err.status || 500).json({
    error: {
      message: err.message || 'Internal server error',
      statusCode: err.status || 500
    }
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: 'Endpoint not found',
      statusCode: 404
    }
  });
});

// Graceful shutdown
const shutdown = async () => {
  logger.info('Shutting down gracefully...');
  
  try {
    await emailClient.disconnectAll();
    logger.info('Email client disconnected');
  } catch (error: any) {
    logger.error('Error disconnecting email client', error);
  }
  
  try {
    await cacheService.disconnect();
    logger.info('Cache service disconnected');
  } catch (error: any) {
    logger.error('Error disconnecting cache', error);
  }
  
  try {
    await disconnectPostgreSQL();
    logger.info('Database disconnected');
  } catch (error: any) {
    logger.error('Error disconnecting database', error);
  }
  
  process.exit(0);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

// Initialize database connection and start server
(async () => {
  try {
    // Connect to database (fail fast if unavailable per requirement)
    await connectPostgreSQL();
    logger.info('Database connection established');
  } catch (error: any) {
    logger.error('Failed to connect to database', error);
    logger.error('Service will fail fast as database is required');
    process.exit(1);
  }
  
  // Start server
  const PORT = config.port;
  app.listen(PORT, () => {
    logger.info(`Extension API service running on port ${PORT}`);
    logger.info(`Environment: ${config.nodeEnv}`);
    logger.info(`Detection API: ${config.detectionApi.url}`);
    logger.info(`Threat Intel: ${config.threatIntel.url}`);
  });
})();
