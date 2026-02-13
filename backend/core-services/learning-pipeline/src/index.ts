import dotenv from 'dotenv';
import { S3Client } from '@aws-sdk/client-s3';
import { ECSClient } from '@aws-sdk/client-ecs';
import { connectPostgreSQL, connectAllDatabases, disconnectAllDatabases } from '../../../shared/database/connection';
import { config } from '../../../shared/config';
import { logger } from './utils/logger';

// Services
import { DataCollectorService } from './services/data-collector.service';
import { FeatureStoreService } from './services/feature-store.service';
import { TrainingOrchestratorService } from './services/training-orchestrator.service';
import { ValidatorService } from './services/validator.service';
import { DriftDetectorService } from './services/drift-detector.service';
import { DeploymentService } from './services/deployment.service';

// Jobs
import { ScheduledTrainingJob } from './jobs/scheduled-training.job';
import { DriftCheckJob } from './jobs/drift-check.job';

dotenv.config();

/**
 * Check if AWS is configured for S3/ECS operations.
 * AWS is considered enabled when AWS_ACCESS_KEY_ID is set, or AWS_ENABLED=true
 */
function isAwsEnabled(): boolean {
  const explicit = process.env.AWS_ENABLED?.toLowerCase();
  if (explicit === 'true' || explicit === '1') return true;
  if (explicit === 'false' || explicit === '0') return false;
  return Boolean(process.env.AWS_ACCESS_KEY_ID);
}

// Global service instances
let dataCollector: DataCollectorService | null = null;
let featureStore: FeatureStoreService | null = null;
let trainingOrchestrator: TrainingOrchestratorService | null = null;
let validator: ValidatorService | null = null;
let driftDetector: DriftDetectorService;
let deployment: DeploymentService | null = null;
let scheduledTrainingJob: ScheduledTrainingJob | null = null;
let driftCheckJob: DriftCheckJob;

/**
 * Initialize all services
 */
async function initializeServices(): Promise<void> {
  try {
    logger.info('Initializing services...');

    // Connect to databases
    logger.info('Connecting to databases...');
    const dataSource = await connectPostgreSQL();
    await connectAllDatabases();
    logger.info('Databases connected');

    // Drift detector is always initialized (DB-backed only)
    // Cast to any to avoid shared/lp typeorm version mismatch
    const ds = dataSource as any;
    driftDetector = new DriftDetectorService(ds);

    const awsEnabled = isAwsEnabled();

    if (awsEnabled) {
      // Initialize AWS clients
      logger.info('Initializing AWS clients...');
      const s3Client = new S3Client({
        region: config.aws.region,
      });

      const ecsClient = new ECSClient({
        region: config.aws.region,
      });
      logger.info('AWS clients initialized');

      // Initialize AWS-dependent services
      dataCollector = new DataCollectorService(ds, s3Client);
      featureStore = new FeatureStoreService(s3Client);
      trainingOrchestrator = new TrainingOrchestratorService(ecsClient, ds);
      validator = new ValidatorService(s3Client, ds);
      deployment = new DeploymentService(s3Client, ecsClient, ds);

      // Initialize scheduled training job (requires AWS)
      scheduledTrainingJob = new ScheduledTrainingJob(
        dataCollector,
        featureStore,
        trainingOrchestrator,
        validator,
        deployment
      );

      // Drift check with optional retraining (requires trainingOrchestrator + dataCollector)
      driftCheckJob = new DriftCheckJob(driftDetector, trainingOrchestrator, dataCollector);

      logger.info('All services initialized successfully (AWS mode)');
    } else {
      // Local mode: no S3/ECS, drift detector only
      logger.info('Learning pipeline running in local mode - S3/ECS disabled');
      logger.info('Set AWS_ACCESS_KEY_ID or AWS_ENABLED=true to enable training pipeline');

      // Drift check without retraining (DB-only)
      driftCheckJob = new DriftCheckJob(driftDetector);

      logger.info('Services initialized (local mode - drift detection only)');
    }
  } catch (error: any) {
    logger.error(`Failed to initialize services: ${error.message}`, error);
    throw error;
  }
}

/**
 * Start scheduled jobs
 */
function startScheduledJobs(): void {
  logger.info('Starting scheduled jobs...');

  if (scheduledTrainingJob) {
    scheduledTrainingJob.start();
  } else {
    logger.info('Scheduled training job skipped (AWS not configured)');
  }

  driftCheckJob.start();

  logger.info('Scheduled jobs started');
}

/**
 * Graceful shutdown
 */
async function shutdown(): Promise<void> {
  logger.info('Shutting down Learning Pipeline service...');

  try {
    await disconnectAllDatabases();
    logger.info('Databases disconnected');
  } catch (error: any) {
    logger.error(`Error during shutdown: ${error.message}`, error);
  }

  process.exit(0);
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  try {
    logger.info('Learning Pipeline service starting...');
    logger.info(`Environment: ${config.environment}`);
    logger.info(`AWS enabled: ${isAwsEnabled()}`);

    if (isAwsEnabled()) {
      logger.info(`AWS Region: ${config.aws.region}`);
      logger.info(`S3 Training Bucket: ${config.aws.s3.training}`);
      logger.info(`S3 Models Bucket: ${config.aws.s3.models}`);
    }

    // Initialize services
    await initializeServices();

    // Start scheduled jobs
    startScheduledJobs();

    // Handle shutdown signals
    process.on('SIGTERM', shutdown);
    process.on('SIGINT', shutdown);

    logger.info('Learning Pipeline service started successfully');
    logger.info(`Training schedule: ${process.env.TRAINING_SCHEDULE || '0 2 * * 0'}`);
    logger.info(`Data collection schedule: ${process.env.DATA_COLLECTION_SCHEDULE || '0 1 * * *'}`);
    logger.info(`Drift check schedule: ${process.env.DRIFT_CHECK_SCHEDULE || '0 */6 * * *'}`);

    // Keep process alive
    setInterval(() => {
      // Health check - log status periodically
      logger.debug('Learning Pipeline service is running');
    }, 60000); // Every minute
  } catch (error: any) {
    logger.error(`Failed to start Learning Pipeline service: ${error.message}`, error);
    process.exit(1);
  }
}

// Start the service
main().catch((error) => {
  logger.error(`Unhandled error: ${error.message}`, error);
  process.exit(1);
});
