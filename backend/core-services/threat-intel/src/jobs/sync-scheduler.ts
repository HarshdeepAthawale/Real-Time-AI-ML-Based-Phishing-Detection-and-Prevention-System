import * as cron from 'node-cron';
import { SyncService } from '../services/sync.service';
import { FeedManagerService } from '../services/feed-manager.service';
import { logger } from '../utils/logger';
import { config } from '../config';

export class SyncScheduler {
  private syncService: SyncService;
  private feedManager: FeedManagerService;
  private jobs: Map<string, cron.ScheduledTask> = new Map();
  
  constructor(syncService: SyncService, feedManager: FeedManagerService) {
    this.syncService = syncService;
    this.feedManager = feedManager;
  }
  
  /**
   * Initialize and start scheduled sync jobs
   */
  async start(): Promise<void> {
    logger.info('Starting sync scheduler');
    
    // Schedule jobs for all active feeds
    await this.scheduleAllFeeds();
    
    // Schedule a periodic job to update feed schedules
    // This checks for new feeds or changed intervals every hour
    cron.schedule('0 * * * *', async () => {
      logger.info('Checking for feed schedule updates');
      await this.scheduleAllFeeds();
    });
    
    logger.info('Sync scheduler started');
  }
  
  /**
   * Schedule sync jobs for all active feeds
   */
  private async scheduleAllFeeds(): Promise<void> {
    try {
      const feeds = await this.feedManager.getActiveFeeds();
      
      // Clear existing jobs
      for (const [feedId, job] of this.jobs.entries()) {
        job.stop();
        this.jobs.delete(feedId);
      }
      
      // Schedule jobs for each feed
      for (const feed of feeds) {
        if (!feed.id || !feed.isActive) {
          continue;
        }
        
        const intervalMinutes = feed.syncIntervalMinutes || config.sync.defaultIntervalMinutes;
        const cronExpression = this.minutesToCronExpression(intervalMinutes);
        
        try {
          const job = cron.schedule(cronExpression, async () => {
            logger.info(`Scheduled sync started for feed: ${feed.name} (${feed.id})`);
            try {
              await this.syncService.syncFeed(feed.id!);
              logger.info(`Scheduled sync completed for feed: ${feed.name} (${feed.id})`);
            } catch (error) {
              logger.error(`Scheduled sync failed for feed: ${feed.name} (${feed.id})`, error);
            }
          }, {
            scheduled: true,
          });
          
          this.jobs.set(feed.id, job);
          logger.info(`Scheduled sync job for feed: ${feed.name} (${feed.id}) - interval: ${intervalMinutes} minutes`);
        } catch (error) {
          logger.error(`Failed to schedule job for feed: ${feed.name} (${feed.id})`, error);
        }
      }
    } catch (error) {
      logger.error('Failed to schedule feed sync jobs', error);
    }
  }
  
  /**
   * Convert minutes to cron expression
   */
  private minutesToCronExpression(minutes: number): string {
    if (minutes < 1) {
      minutes = 1; // Minimum 1 minute
    }
    
    if (minutes >= 60) {
      // If >= 60 minutes, convert to hours
      const hours = Math.floor(minutes / 60);
      if (hours >= 24) {
        // If >= 24 hours, convert to days
        const days = Math.floor(hours / 24);
        return `0 0 */${days} * *`; // Every N days at midnight
      }
      return `0 */${hours} * * *`; // Every N hours at minute 0
    }
    
    // Less than 60 minutes
    return `*/${minutes} * * * *`; // Every N minutes
  }
  
  /**
   * Stop all scheduled jobs
   */
  stop(): void {
    logger.info('Stopping sync scheduler');
    
    for (const [feedId, job] of this.jobs.entries()) {
      job.stop();
      this.jobs.delete(feedId);
    }
    
    logger.info('Sync scheduler stopped');
  }
  
  /**
   * Get status of scheduled jobs
   */
  getStatus(): { feedId: string; scheduled: boolean }[] {
    const status: { feedId: string; scheduled: boolean }[] = [];
    
    for (const feedId of this.jobs.keys()) {
      status.push({
        feedId,
        scheduled: true,
      });
    }
    
    return status;
  }
}
