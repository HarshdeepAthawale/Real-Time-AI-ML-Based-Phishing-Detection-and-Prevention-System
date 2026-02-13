import { MISPClient } from '../integrations/misp.client';
import { OTXClient } from '../integrations/otx.client';
import { BaseFeedClient } from '../integrations/base-feed.client';
import { IOCManagerService } from './ioc-manager.service';
import { IOCMatcherService } from './ioc-matcher.service';
import { FeedManagerService, Feed } from './feed-manager.service';
import { logger } from '../utils/logger';
import { IOC } from '../models/ioc.model';

export interface SyncResult {
  feedId: string;
  feedName: string;
  success: boolean;
  iocsInserted: number;
  error?: string;
}

export class SyncService {
  private iocManager: IOCManagerService;
  private iocMatcher: IOCMatcherService;
  private feedManager: FeedManagerService;
  private feedClients: Map<string, BaseFeedClient> = new Map();
  
  constructor(
    iocManager: IOCManagerService,
    iocMatcher: IOCMatcherService,
    feedManager: FeedManagerService
  ) {
    this.iocManager = iocManager;
    this.iocMatcher = iocMatcher;
    this.feedManager = feedManager;
  }
  
  /**
   * Initialize feed clients from configured feeds
   */
  async initializeFeedClients(): Promise<void> {
    const feeds = await this.feedManager.getAllFeeds();
    
    for (const feed of feeds) {
      if (!feed.isActive) {
        continue;
      }
      
      try {
        const client = await this.createFeedClient(feed);
        if (client) {
          this.feedClients.set(feed.id!, client);
        }
      } catch (error) {
        logger.error(`Failed to initialize feed client for ${feed.name}`, error);
      }
    }
    
    logger.info(`Initialized ${this.feedClients.size} feed clients`);
  }
  
  /**
   * Create feed client based on feed type
   */
  private async createFeedClient(feed: Feed): Promise<BaseFeedClient | undefined> {
    switch (feed.feedType) {
      case 'misp':
        if (feed.apiEndpoint && feed.apiKeyEncrypted) {
          // In production, decrypt apiKeyEncrypted
          return new MISPClient(feed.apiEndpoint, feed.apiKeyEncrypted) as unknown as BaseFeedClient;
        }
        break;
      
      case 'otx':
        if (feed.apiKeyEncrypted) {
          // In production, decrypt apiKeyEncrypted
          return new OTXClient(feed.apiKeyEncrypted) as unknown as BaseFeedClient;
        }
        break;
      
      default:
        logger.warn(`Unsupported feed type: ${feed.feedType}`);
    }
    
    return undefined;
  }
  
  /**
   * Sync all active feeds
   */
  async syncAllFeeds(): Promise<SyncResult[]> {
    try {
      const activeFeeds = await this.feedManager.getActiveFeeds();
      
      if (activeFeeds.length === 0) {
        logger.info('No active feeds to sync');
        return [];
      }

      const results: SyncResult[] = [];
      
      for (const feed of activeFeeds) {
        if (!feed.id) {
          logger.warn(`Feed ${feed.name} has no ID, skipping`);
          continue;
        }

        try {
          const result = await this.syncFeed(feed.id);
          results.push(result);
        } catch (error: any) {
          logger.error(`Failed to sync feed ${feed.name}`, error);
          results.push({
            feedId: feed.id,
            feedName: feed.name,
            success: false,
            iocsInserted: 0,
            error: error?.message || String(error),
          });
        }
      }
      
      return results;
    } catch (error) {
      logger.error('Failed to sync all feeds', error);
      throw error;
    }
  }
  
  /**
   * Sync a specific feed
   */
  async syncFeed(feedId: string): Promise<SyncResult> {
    if (!feedId) {
      throw new Error('Feed ID is required');
    }

    const feed = await this.feedManager.getFeedById(feedId);
    if (!feed) {
      throw new Error(`Feed with id ${feedId} not found`);
    }

    try {
      
      if (!feed.isActive) {
        throw new Error(`Feed ${feed.name} is not active`);
      }
      
      let client = this.feedClients.get(feedId);
      
      // Create client if not already initialized
      if (!client) {
        const newClient = await this.createFeedClient(feed);
        if (!newClient) {
          throw new Error(`Failed to create client for feed ${feed.name}. Check feed configuration (API endpoint/key).`);
        }
        client = newClient;
        this.feedClients.set(feedId, client);
      }
      
      // Get last sync time
      const lastSyncAt = feed.lastSyncAt || undefined;
      
      // Fetch IOCs from feed
      logger.info(`Syncing feed ${feed.name} (since: ${lastSyncAt || 'beginning'})`);
      const iocs = await client.fetchIOCs(lastSyncAt);
      
      if (!iocs || iocs.length === 0) {
        await this.feedManager.updateSyncStatus(feedId, 'success');
        logger.info(`Sync completed for ${feed.name}: no new IOCs`);
        return {
          feedId,
          feedName: feed.name,
          success: true,
          iocsInserted: 0,
        };
      }
      
      // Validate IOCs before processing
      const validIOCs = iocs.filter(ioc => {
        if (!ioc.iocType || !ioc.iocValue) {
          logger.warn(`Skipping invalid IOC from feed ${feed.name}: missing type or value`, { ioc });
          return false;
        }
        return true;
      });

      if (validIOCs.length === 0) {
        logger.warn(`No valid IOCs found in feed ${feed.name}`);
        await this.feedManager.updateSyncStatus(feedId, 'partial', 'No valid IOCs found');
        return {
          feedId,
          feedName: feed.name,
          success: true,
          iocsInserted: 0,
        };
      }
      
      // Add feed ID to IOCs
      const iocsWithFeedId: IOC[] = validIOCs.map(ioc => ({
        ...ioc,
        feedId,
      }));
      
      // Bulk insert IOCs
      const inserted = await this.iocManager.bulkCreateIOCs(iocsWithFeedId);
      
      // Update bloom filters
      await this.updateBloomFilters(validIOCs);
      
      // Update sync status
      await this.feedManager.updateSyncStatus(
        feedId,
        inserted === validIOCs.length ? 'success' : 'partial'
      );
      
      logger.info(`Sync completed for ${feed.name}: ${inserted}/${validIOCs.length} IOCs inserted`);
      
      return {
        feedId,
        feedName: feed.name,
        success: true,
        iocsInserted: inserted,
      };
    } catch (error: any) {
      const errorMessage = error?.message || String(error);
      logger.error(`Sync failed for feed ${feedId}`, error);
      
      // If feed not found, throw error to be handled by route (404)
      if (errorMessage.includes('not found')) {
        throw error;
      }
      
      // Update sync status with error for other errors
      try {
        await this.feedManager.updateSyncStatus(
          feedId,
          'failed',
          errorMessage
        );
      } catch (updateError) {
        logger.error('Failed to update sync status', updateError);
      }
      
      return {
        feedId,
        feedName: 'unknown',
        success: false,
        iocsInserted: 0,
        error: errorMessage,
      };
    }
  }
  
  /**
   * Update bloom filters with new IOCs
   */
  private async updateBloomFilters(iocs: IOC[]): Promise<void> {
    if (!iocs || iocs.length === 0) {
      return;
    }

    // Group IOCs by type
    const iocsByType = new Map<string, string[]>();
    
    for (const ioc of iocs) {
      if (!ioc.iocType || !ioc.iocValue) {
        continue; // Skip invalid IOCs
      }

      if (!iocsByType.has(ioc.iocType)) {
        iocsByType.set(ioc.iocType, []);
      }
      iocsByType.get(ioc.iocType)!.push(ioc.iocValue);
    }
    
    // Update bloom filters for each type
    for (const [iocType, values] of iocsByType.entries()) {
      if (values.length === 0) {
        continue;
      }

      try {
        await this.iocMatcher.bulkAddToBloomFilter(
          iocType as any,
          values
        );
      } catch (error) {
        logger.error(`Failed to update bloom filter for ${iocType}`, error);
        // Don't throw - bloom filter update failure shouldn't fail the sync
      }
    }
  }
  
  /**
   * Get sync status for all feeds
   */
  async getSyncStatus(): Promise<{
    feeds: Feed[];
    totalFeeds: number;
    activeFeeds: number;
    lastSyncOverall?: Date;
  }> {
    const feeds = await this.feedManager.getAllFeeds();
    const activeFeeds = feeds.filter(f => f.isActive);
    
    // Find most recent sync
    let lastSyncOverall: Date | undefined;
    for (const feed of feeds) {
      if (feed.lastSyncAt) {
        if (!lastSyncOverall || feed.lastSyncAt > lastSyncOverall) {
          lastSyncOverall = feed.lastSyncAt;
        }
      }
    }
    
    return {
      feeds,
      totalFeeds: feeds.length,
      activeFeeds: activeFeeds.length,
      lastSyncOverall,
    };
  }
}
