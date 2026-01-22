import { IntelligenceService } from '../services/intelligence.service';
import { logger } from '../utils/logger';
import * as cron from 'node-cron';

export class FeedSyncJob {
  private intelligenceService: IntelligenceService;
  private syncIntervalMinutes: number;
  private cronJob: cron.ScheduledTask | null = null;

  constructor(intelligenceService: IntelligenceService) {
    this.intelligenceService = intelligenceService;
    this.syncIntervalMinutes = parseInt(process.env.SYNC_INTERVAL_MINUTES || '60');
  }

  /**
   * Start the feed sync scheduler
   */
  start(): void {
    // Convert minutes to cron expression
    // Run every N minutes
    const cronExpression = `*/${this.syncIntervalMinutes} * * * *`;

    logger.info(`Starting feed sync scheduler: every ${this.syncIntervalMinutes} minutes`);

    this.cronJob = cron.schedule(cronExpression, async () => {
      try {
        logger.info('Running scheduled feed synchronization...');
        
        const result = await this.intelligenceService.syncAllFeeds();
        
        if (result.success) {
          logger.info(`Feed sync successful: ${result.feeds_synced.length} feeds, ${result.total_iocs} IOCs`);
        } else {
          logger.warn(`Feed sync completed with errors: ${result.errors.join(', ')}`);
        }
      } catch (error: any) {
        logger.error(`Scheduled feed sync failed: ${error.message}`);
      }
    });

    // Run initial sync
    this.runInitialSync();

    logger.info('Feed sync scheduler started');
  }

  /**
   * Stop the scheduler
   */
  stop(): void {
    if (this.cronJob) {
      this.cronJob.stop();
      logger.info('Feed sync scheduler stopped');
    }
  }

  /**
   * Run initial sync on startup
   */
  private async runInitialSync(): Promise<void> {
    try {
      logger.info('Running initial feed synchronization...');
      
      // Wait a bit for services to fully initialize
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      const result = await this.intelligenceService.syncAllFeeds();
      
      if (result.success) {
        logger.info(`Initial sync successful: ${result.feeds_synced.length} feeds, ${result.total_iocs} IOCs`);
      } else {
        logger.warn(`Initial sync completed with errors: ${result.errors.join(', ')}`);
      }
    } catch (error: any) {
      logger.error(`Initial feed sync failed: ${error.message}`);
    }
  }

  /**
   * Manually trigger sync
   */
  async triggerSync(): Promise<void> {
    try {
      logger.info('Manually triggering feed synchronization...');
      await this.intelligenceService.syncAllFeeds();
    } catch (error: any) {
      logger.error(`Manual feed sync failed: ${error.message}`);
      throw error;
    }
  }
}
