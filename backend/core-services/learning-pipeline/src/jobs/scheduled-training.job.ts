import cron from 'node-cron';
import { DataCollectorService } from '../services/data-collector.service';
import { FeatureStoreService } from '../services/feature-store.service';
import { TrainingOrchestratorService } from '../services/training-orchestrator.service';
import { ValidatorService } from '../services/validator.service';
import { DeploymentService } from '../services/deployment.service';
import { logger } from '../utils/logger';

export class ScheduledTrainingJob {
  private dataCollector: DataCollectorService;
  private featureStore: FeatureStoreService;
  private trainingOrchestrator: TrainingOrchestratorService;
  private validator: ValidatorService;
  private deployment: DeploymentService;

  constructor(
    dataCollector: DataCollectorService,
    featureStore: FeatureStoreService,
    trainingOrchestrator: TrainingOrchestratorService,
    validator: ValidatorService,
    deployment: DeploymentService
  ) {
    this.dataCollector = dataCollector;
    this.featureStore = featureStore;
    this.trainingOrchestrator = trainingOrchestrator;
    this.validator = validator;
    this.deployment = deployment;
  }

  /**
   * Start scheduled jobs
   */
  start(): void {
    // Daily data collection (default: 1 AM)
    const dataCollectionSchedule = process.env.DATA_COLLECTION_SCHEDULE || '0 1 * * *';
    cron.schedule(dataCollectionSchedule, async () => {
      logger.info('Scheduled data collection job triggered');
      try {
        const since = new Date();
        since.setDate(since.getDate() - 1); // Collect data from last 24 hours

        const results = await this.dataCollector.collectAllTrainingData(since);
        logger.info(`Data collection completed: ${JSON.stringify(results)}`);
      } catch (error: any) {
        logger.error(`Scheduled data collection failed: ${error.message}`, error);
      }
    });

    // Weekly training (default: Sunday at 2 AM)
    const trainingSchedule = process.env.TRAINING_SCHEDULE || '0 2 * * 0';
    cron.schedule(trainingSchedule, async () => {
      logger.info('Scheduled training job triggered');
      try {
        await this.runTrainingPipeline('nlp');
        await this.runTrainingPipeline('url');
        await this.runTrainingPipeline('visual');
      } catch (error: any) {
        logger.error(`Scheduled training failed: ${error.message}`, error);
      }
    });

    logger.info(`Scheduled jobs started - Data collection: ${dataCollectionSchedule}, Training: ${trainingSchedule}`);
  }

  /**
   * Run full training pipeline for a model type
   */
  private async runTrainingPipeline(modelType: string): Promise<void> {
    try {
      logger.info(`Starting training pipeline for ${modelType}`);

      // 1. Prepare dataset
      const datasetPath = await this.featureStore.prepareDataset(modelType);
      logger.info(`Dataset prepared: ${datasetPath}`);

      // 2. Trigger training
      const jobId = await this.trainingOrchestrator.triggerTraining({
        modelType,
        datasetPath,
        trainingConfig: {
          epochs: 3,
          batchSize: 16,
          learningRate: 2e-5,
        },
      });
      logger.info(`Training job started: ${jobId} for model ${modelType}`);

      // Note: In production, you would:
      // 3. Poll for training completion (or use event-driven approach)
      // 4. Download and validate the trained model
      // 5. Compare with current production model
      // 6. Deploy if better

      // For now, we'll log that training has been triggered
      // The actual validation and deployment would happen in a separate process
      // that monitors training job completion

    } catch (error: any) {
      logger.error(`Training pipeline failed for ${modelType}: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Manually trigger training for a model type
   */
  async triggerTraining(modelType: string): Promise<void> {
    await this.runTrainingPipeline(modelType);
  }
}
