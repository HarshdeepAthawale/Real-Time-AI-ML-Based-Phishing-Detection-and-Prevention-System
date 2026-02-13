import 'reflect-metadata';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import { config } from './config';
import { logger } from './utils/logger';
import { connectDatabase, disconnectDatabase, getDatabase } from './services/database.service';
import { connectRedis, disconnectRedis, isRedisConnected, getRedis } from './services/redis.service';
import { CuckooClient } from './integrations/cuckoo.client';
import { AnyRunClient } from './integrations/anyrun.client';
import { CuckooSandboxAdapter, AnyRunSandboxAdapter, DisabledSandboxAdapter } from './integrations/sandbox-adapters';
import { BaseSandboxClient } from './integrations/base-sandbox.client';
import { FileAnalyzerService } from './services/file-analyzer.service';
import { SandboxSubmitterService } from './services/sandbox-submitter.service';
import { ResultProcessorService } from './services/result-processor.service';
import { BehavioralAnalyzerService } from './services/behavioral-analyzer.service';
import { CorrelationService } from './services/correlation.service';
import { SandboxQueueJob } from './jobs/sandbox-queue.job';
import { setupSandboxRoutes } from './routes/sandbox.routes';

const app = express();
const PORT = process.env.PORT || 3004;

// Middleware
app.use(helmet());
app.use(cors(config.cors));
app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Initialize services (will be set after database connection)
let sandboxClient: BaseSandboxClient;
let fileAnalyzer: FileAnalyzerService;
let submitterService: SandboxSubmitterService;
let resultProcessorService: ResultProcessorService;
let behavioralAnalyzer: BehavioralAnalyzerService;
let correlationService: CorrelationService;
let queueJob: SandboxQueueJob;

// Initialize application
async function initializeApp(): Promise<void> {
  try {
    // Step 1: Connect to database
    logger.info('Connecting to database...');
    const dataSource = await connectDatabase();
    
    // Step 2: Connect to Redis
    logger.info('Connecting to Redis...');
    const redis = await connectRedis();
    
    // Step 3: Initialize sandbox client based on configuration
    logger.info(`Initializing sandbox client: ${config.provider}`);
    if (config.provider === 'cuckoo') {
      const cuckoo = new CuckooClient();
      sandboxClient = new CuckooSandboxAdapter(cuckoo);
      logger.info('Cuckoo Sandbox client initialized');
    } else if (config.provider === 'anyrun') {
      const anyrun = new AnyRunClient();
      sandboxClient = new AnyRunSandboxAdapter(anyrun);
      logger.info('Any.run client initialized');
    } else if (config.provider === 'disabled') {
      sandboxClient = new DisabledSandboxAdapter();
      logger.warn('Sandbox analysis DISABLED – no provider configured. Set ANYRUN_API_KEY or CUCKOO_SANDBOX_URL to enable.');
    } else {
      throw new Error(`Unknown sandbox provider: ${config.provider}`);
    }
    
    // Step 4: Initialize core services
    logger.info('Initializing services...');
    fileAnalyzer = new FileAnalyzerService();
    behavioralAnalyzer = new BehavioralAnalyzerService();
    submitterService = new SandboxSubmitterService(sandboxClient, fileAnalyzer, dataSource);
    correlationService = new CorrelationService(
      dataSource,
      process.env.DETECTION_API_URL
    );
    resultProcessorService = new ResultProcessorService(
      sandboxClient,
      dataSource,
      behavioralAnalyzer,
      correlationService
    );
    
    // Step 5: Initialize job queue
    logger.info('Initializing job queue...');
    queueJob = new SandboxQueueJob(redis, resultProcessorService);
    logger.info('Job queue initialized');
    
    // Step 6: Set up routes
    logger.info('Setting up routes...');
    const sandboxRoutes = setupSandboxRoutes(
      submitterService,
      resultProcessorService,
      queueJob,
      dataSource,
      config
    );
    app.use('/api/v1/sandbox', sandboxRoutes);
    
    logger.info('Application initialized successfully');
  } catch (error: any) {
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

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    let dbConnected = false;
    try {
      dbConnected = getDatabase().isInitialized;
    } catch {
      dbConnected = false;
    }
    const redisConnected = isRedisConnected();
    const isDisabled = config.provider === 'disabled';
    
    const health = {
      status: isDisabled ? 'degraded' : (dbConnected && redisConnected ? 'healthy' : 'degraded'),
      service: 'sandbox-service',
      provider: config.provider,
      sandboxEnabled: !isDisabled,
      database: dbConnected ? 'connected' : 'disconnected',
      redis: redisConnected ? 'connected' : 'disconnected',
      timestamp: new Date().toISOString(),
    };
    
    // Report healthy even when disabled (service is running; analysis is just unavailable)
    const isHealthy = dbConnected && redisConnected;
    res.status(isHealthy ? 200 : 503).json(health);
  } catch (error: any) {
    logger.error('Health check error', error);
    res.status(503).json({
      status: 'unhealthy',
      service: 'sandbox-service',
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});

// Graceful shutdown
async function shutdown(): Promise<void> {
  logger.info('Shutting down gracefully...');
  
  try {
    if (queueJob) {
      await queueJob.close();
    }
  } catch (error) {
    logger.error('Error closing queue', error);
  }
  
  try {
    await disconnectRedis();
  } catch (error) {
    logger.error('Error disconnecting Redis', error);
  }
  
  try {
    await disconnectDatabase();
  } catch (error) {
    logger.error('Error disconnecting database', error);
  }
  
  logger.info('Shutdown complete');
}

// Handle shutdown signals
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received');
  await shutdown();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received');
  await shutdown();
  process.exit(0);
});

// Start server
initializeApp()
  .then(() => {
    app.listen(PORT, () => {
      logger.info(`Sandbox service running on port ${PORT}`);
      logger.info(`Provider: ${config.provider}`);
    });
  })
  .catch((error) => {
    logger.error('Failed to start server', error);
    process.exit(1);
  });
