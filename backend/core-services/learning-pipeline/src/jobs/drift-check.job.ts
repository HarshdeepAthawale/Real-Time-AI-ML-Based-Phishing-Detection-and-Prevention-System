import cron from 'node-cron';
import { DriftDetectorService } from '../services/drift-detector.service';
import { logger } from '../utils/logger';

export class DriftCheckJob {
  private driftDetector: DriftDetectorService;

  constructor(driftDetector: DriftDetectorService) {
    this.driftDetector = driftDetector;
  }

  /**
   * Start scheduled drift checks
   */
  start(): void {
    // Periodic drift check (default: every 6 hours)
    const driftCheckSchedule = process.env.DRIFT_CHECK_SCHEDULE || '0 */6 * * *';
    
    cron.schedule(driftCheckSchedule, async () => {
      logger.info('Scheduled drift check job triggered');
      try {
        const results = await this.driftDetector.checkAllModels(7); // 7 day window

        let driftCount = 0;
        for (const [modelId, result] of results.entries()) {
          if (result.hasDrift) {
            driftCount++;
            logger.warn(`Drift detected for model ${modelId}: ${result.recommendation}`);
          }
        }

        if (driftCount === 0) {
          logger.info('No drift detected in any models');
        } else {
          logger.warn(`Drift detected in ${driftCount} model(s). Consider retraining.`);
        }
      } catch (error: any) {
        logger.error(`Scheduled drift check failed: ${error.message}`, error);
      }
    });

    logger.info(`Drift check job started - Schedule: ${driftCheckSchedule}`);
  }

  /**
   * Manually trigger drift check
   */
  async checkDrift(windowDays: number = 7): Promise<void> {
    logger.info(`Manual drift check triggered (window: ${windowDays} days)`);
    const results = await this.driftDetector.checkAllModels(windowDays);

    for (const [modelId, result] of results.entries()) {
      if (result.hasDrift) {
        logger.warn(`Drift detected for model ${modelId}: ${result.recommendation}`);
      }
    }
  }
}
