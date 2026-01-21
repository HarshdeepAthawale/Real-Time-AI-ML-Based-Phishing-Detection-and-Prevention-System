import 'reflect-metadata';
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import { config } from './config';
import { logger } from './utils/logger';
import { errorHandler } from './middleware/error-handler.middleware';
import { connectDatabase, disconnectDatabase, getDatabase } from './services/database.service';
import { connectRedis, disconnectRedis, isRedisConnected, getRedis } from './services/redis.service';
import { IOCManagerService } from './services/ioc-manager.service';
import { IOCMatcherService } from './services/ioc-matcher.service';
import { FeedManagerService } from './services/feed-manager.service';
import { SyncService } from './services/sync.service';
import { EnrichmentService } from './services/enrichment.service';
import { SyncScheduler } from './jobs/sync-scheduler';
import iocRoutes from './routes/ioc.routes';
import feedsRoutes from './routes/feeds.routes';
import syncRoutes from './routes/sync.routes';

const app = express();

// Middleware
app.use(helmet());
app.use(cors(config.cors));
app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Initialize services (will be set after database connection)
let iocManager: IOCManagerService;
let iocMatcher: IOCMatcherService;
let feedManager: FeedManagerService;
let syncService: SyncService;
let enrichmentService: EnrichmentService;
let syncScheduler: SyncScheduler;

// Initialize application
let dataSource: ReturnType<typeof getDatabase> | null = null;

async function initializeApp(): Promise<void> {
  try {
    // Step 1: Connect to database
    logger.info('Connecting to database...');
    dataSource = await connectDatabase();
    
    // Step 2: Connect to Redis
    logger.info('Connecting to Redis...');
    const redis = await connectRedis();
    
    // Step 3: Initialize core services (order matters)
    logger.info('Initializing services...');
    iocManager = new IOCManagerService(dataSource!);
    iocMatcher = new IOCMatcherService(redis, iocManager);
    feedManager = new FeedManagerService(dataSource!);
    enrichmentService = new EnrichmentService(iocManager);
    syncService = new SyncService(iocManager, iocMatcher, feedManager);
    syncScheduler = new SyncScheduler(syncService, feedManager);
    
    // Step 4: Initialize bloom filters (requires IOC matcher)
    logger.info('Initializing bloom filters...');
    try {
      await iocMatcher.initializeBloomFilters();
      logger.info('Bloom filters initialized');
    } catch (error) {
      logger.error('Failed to initialize bloom filters (non-critical)', error);
      // Continue - bloom filters can be rebuilt later
    }
    
    // Step 5: Initialize feed clients (requires feed manager)
    logger.info('Initializing feed clients...');
    try {
      await syncService.initializeFeedClients();
      logger.info('Feed clients initialized');
    } catch (error) {
      logger.error('Failed to initialize some feed clients (non-critical)', error);
      // Continue - feeds can be configured later
    }
    
    // Step 6: Start sync scheduler (requires sync service)
    logger.info('Starting sync scheduler...');
    try {
      await syncScheduler.start();
      logger.info('Sync scheduler started');
    } catch (error) {
      logger.error('Failed to start sync scheduler (non-critical)', error);
      // Continue - scheduler can be started manually later
    }
    
    // Step 7: Set services in app for routes
    app.set('iocManager', iocManager);
    app.set('iocMatcher', iocMatcher);
    app.set('feedManager', feedManager);
    app.set('syncService', syncService);
    app.set('enrichmentService', enrichmentService);
    
    logger.info('Application initialized successfully');
  } catch (error) {
    logger.error('Failed to initialize application', error);
    // Attempt graceful cleanup before exiting
    try {
      await shutdown();
    } catch (cleanupError) {
      logger.error('Error during cleanup', cleanupError);
    }
    process.exit(1);
  }
}

// Routes
app.use('/api/v1/ioc', iocRoutes);
app.use('/api/v1/feeds', feedsRoutes);
app.use('/api/v1/sync', syncRoutes);

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    let dbConnected = false;
    try {
      const db = getDatabase();
      dbConnected = db.isInitialized;
    } catch {
      dbConnected = false;
    }
    const redisConnected = isRedisConnected();
    
    const health = {
      status: dbConnected && redisConnected ? 'healthy' : 'degraded',
      service: 'threat-intel',
      database: dbConnected ? 'connected' : 'disconnected',
      redis: redisConnected ? 'connected' : 'disconnected',
      timestamp: new Date().toISOString(),
    };
    
    const isHealthy = dbConnected && redisConnected;
    res.status(isHealthy ? 200 : 503).json(health);
  } catch (error) {
    logger.error('Health check error', error);
    res.status(503).json({
      status: 'unhealthy',
      service: 'threat-intel',
      error: (error as Error).message,
      timestamp: new Date().toISOString(),
    });
  }
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    service: 'Threat Intelligence Service',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      ioc: '/api/v1/ioc',
      feeds: '/api/v1/feeds',
      sync: '/api/v1/sync',
    },
  });
});

// Error handling
app.use(errorHandler);

// Graceful shutdown
async function shutdown(): Promise<void> {
  logger.info('Shutting down gracefully...');
  
  const shutdownTimeout = setTimeout(() => {
    logger.error('Shutdown timeout - forcing exit');
    process.exit(1);
  }, 30000); // 30 second timeout
  
  try {
    // Step 1: Stop accepting new requests (Express handles this automatically)
    logger.info('Stopping sync scheduler...');
    if (syncScheduler) {
      try {
        syncScheduler.stop();
        logger.info('Sync scheduler stopped');
      } catch (error) {
        logger.error('Error stopping sync scheduler', error);
      }
    }
    
    // Step 2: Disconnect Redis (allows pending operations to complete)
    logger.info('Disconnecting Redis...');
    try {
      await disconnectRedis();
      logger.info('Redis disconnected');
    } catch (error) {
      logger.error('Error disconnecting Redis', error);
    }
    
    // Step 3: Disconnect database (allows pending queries to complete)
    logger.info('Disconnecting database...');
    try {
      await disconnectDatabase();
      logger.info('Database disconnected');
    } catch (error) {
      logger.error('Error disconnecting database', error);
    }
    
    clearTimeout(shutdownTimeout);
    logger.info('Shutdown complete');
    process.exit(0);
  } catch (error) {
    clearTimeout(shutdownTimeout);
    logger.error('Error during shutdown', error);
    process.exit(1);
  }
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

// Start server
const PORT = config.port;

// Initialize and start
initializeApp()
  .then(() => {
    app.listen(PORT, () => {
      logger.info(`Threat Intelligence service running on port ${PORT}`);
      logger.info(`Environment: ${config.nodeEnv}`);
    });
  })
  .catch((error) => {
    logger.error('Failed to start server', error);
    process.exit(1);
  });
