import cron from 'node-cron';
import { DriftDetectorService, DriftResult } from '../services/drift-detector.service';
import { TrainingOrchestratorService } from '../services/training-orchestrator.service';
import { DataCollectorService } from '../services/data-collector.service';
import { logger } from '../utils/logger';

export class DriftCheckJob {
  private driftDetector: DriftDetectorService;
  private trainingOrchestrator?: TrainingOrchestratorService;
  private dataCollector?: DataCollectorService;
  private autoRetrain: boolean;

  constructor(
    driftDetector: DriftDetectorService,
    trainingOrchestrator?: TrainingOrchestratorService,
    dataCollector?: DataCollectorService
  ) {
    this.driftDetector = driftDetector;
    this.trainingOrchestrator = trainingOrchestrator;
    this.dataCollector = dataCollector;
    this.autoRetrain = process.env.AUTO_RETRAIN_ON_DRIFT === 'true';
  }

  /**
   * Start scheduled drift checks
   */
  start(): void {
    const driftCheckSchedule = process.env.DRIFT_CHECK_SCHEDULE || '0 */6 * * *';

    cron.schedule(driftCheckSchedule, async () => {
      logger.info('Scheduled drift check job triggered');
      await this.runDriftCheckAndRetrain(7);
    });

    logger.info(`Drift check job started - Schedule: ${driftCheckSchedule}, Auto-retrain: ${this.autoRetrain}`);
  }

  /**
   * Run full drift check -> data collection -> retraining pipeline
   */
  async runDriftCheckAndRetrain(windowDays: number = 7): Promise<{
    driftResults: Map<string, DriftResult>;
    retrainingTriggered: string[];
  }> {
    const retrainingTriggered: string[] = [];

    try {
      // Step 1: Check drift for all active models
      const results = await this.driftDetector.checkAllModels(windowDays);

      let driftCount = 0;
      const driftedModels: { modelId: string; result: DriftResult }[] = [];

      for (const [modelId, result] of results.entries()) {
        if (result.hasDrift) {
          driftCount++;
          driftedModels.push({ modelId, result });
          logger.warn(`Drift detected for model ${modelId}`, {
            driftScore: result.driftScore,
            recommendation: result.recommendation,
            recentF1: result.metrics.recent.f1Score,
            baselineF1: result.metrics.baseline.f1Score,
            recentFPR: result.metrics.recent.falsePositiveRate,
            baselineFPR: result.metrics.baseline.falsePositiveRate,
          });
        }
      }

      if (driftCount === 0) {
        logger.info('No drift detected in any models');
        return { driftResults: results, retrainingTriggered: [] };
      }

      logger.warn(`Drift detected in ${driftCount} model(s)`);

      // Step 2: If auto-retrain enabled, collect data and trigger retraining
      if (this.autoRetrain && this.trainingOrchestrator && this.dataCollector) {
        // Collect fresh training data (feedback + threats + false positives)
        const since = new Date();
        since.setDate(since.getDate() - 30); // Last 30 days of data

        logger.info('Collecting training data for retraining...');
        const collectionResult = await this.dataCollector.collectAllTrainingData(since);

        logger.info('Training data collected', {
          feedback: collectionResult.feedback,
          threats: collectionResult.threats,
          falsePositives: collectionResult.falsePositives,
        });

        // Step 3: Trigger retraining for each drifted model
        for (const { modelId, result } of driftedModels) {
          try {
            const modelType = this.inferModelType(modelId);
            if (!modelType) {
              logger.warn(`Cannot determine model type for ${modelId}, skipping retraining`);
              continue;
            }

            const timestamp = Date.now();
            const datasetPath = `s3://phishing-detection-training/training-data/combined/${timestamp}`;

            logger.info(`Triggering retraining for model ${modelId} (type: ${modelType})`, {
              driftScore: result.driftScore,
              datasetPath,
            });

            const jobId = await this.trainingOrchestrator.triggerTraining({
              modelType,
              datasetPath,
              trainingConfig: {
                reason: 'drift_detected',
                driftScore: result.driftScore,
                previousF1: result.metrics.baseline.f1Score,
                currentF1: result.metrics.recent.f1Score,
                triggerTimestamp: new Date().toISOString(),
              },
            });

            retrainingTriggered.push(modelId);
            logger.info(`Retraining job ${jobId} started for model ${modelId}`);
          } catch (error: any) {
            logger.error(`Failed to trigger retraining for model ${modelId}`, {
              error: error.message,
            });
          }
        }
      } else if (driftCount > 0) {
        logger.info('Auto-retrain is disabled. Manual retraining required for drifted models.');
      }

      return { driftResults: results, retrainingTriggered };
    } catch (error: any) {
      logger.error(`Drift check and retrain pipeline failed: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Manually trigger drift check
   */
  async checkDrift(windowDays: number = 7): Promise<void> {
    logger.info(`Manual drift check triggered (window: ${windowDays} days)`);
    const { driftResults, retrainingTriggered } = await this.runDriftCheckAndRetrain(windowDays);

    for (const [modelId, result] of driftResults.entries()) {
      if (result.hasDrift) {
        logger.warn(`Drift detected for model ${modelId}: ${result.recommendation}`);
      }
    }

    if (retrainingTriggered.length > 0) {
      logger.info(`Retraining triggered for ${retrainingTriggered.length} model(s)`);
    }
  }

  /**
   * Infer model type from model ID or name
   */
  private inferModelType(modelId: string): string | null {
    const lower = modelId.toLowerCase();
    if (lower.includes('nlp') || lower.includes('phishing-classifier') || lower.includes('text')) {
      return 'nlp';
    }
    if (lower.includes('url') || lower.includes('gnn') || lower.includes('domain')) {
      return 'url';
    }
    if (lower.includes('visual') || lower.includes('cnn') || lower.includes('image')) {
      return 'visual';
    }
    return null;
  }
}
