import { DataSource, Repository, QueryRunner } from 'typeorm';
import { ThreatIntelligenceFeed as FeedEntity } from '../../../../shared/database/models/ThreatIntelligenceFeed';
import { logger } from '../utils/logger';

export interface Feed {
  id?: string;
  name: string;
  feedType: 'misp' | 'otx' | 'custom' | 'user_submitted';
  apiEndpoint?: string | null;
  apiKeyEncrypted?: string | null;
  syncIntervalMinutes?: number;
  lastSyncAt?: Date | null;
  lastSyncStatus?: 'success' | 'failed' | 'partial' | null;
  lastSyncError?: string | null;
  isActive?: boolean;
  reliabilityScore?: number;
  createdAt?: Date;
  updatedAt?: Date;
}

export class FeedManagerService {
  private feedRepository: Repository<FeedEntity>;
  
  constructor(private dataSource: DataSource, private queryRunner?: QueryRunner) {
    // Use query runner's manager if available (for transaction isolation in tests)
    // Otherwise use dataSource manager (normal operation)
    const manager = queryRunner?.manager || dataSource.manager || dataSource;
    this.feedRepository = manager.getRepository(FeedEntity);
  }
  
  /**
   * Create a new feed
   */
  async createFeed(feed: Omit<Feed, 'id' | 'createdAt' | 'updatedAt'>): Promise<Feed> {
    try {
      const newFeed = this.feedRepository.create({
        name: feed.name,
        feed_type: feed.feedType,
        api_endpoint: feed.apiEndpoint || null,
        api_key_encrypted: feed.apiKeyEncrypted || null,
        sync_interval_minutes: feed.syncIntervalMinutes || 60,
        is_active: feed.isActive !== undefined ? feed.isActive : true,
        reliability_score: feed.reliabilityScore || 50,
      });
      
      const saved = await this.feedRepository.save(newFeed);
      return this.mapEntityToDTO(saved);
    } catch (error) {
      logger.error('Failed to create feed', error);
      throw error;
    }
  }
  
  /**
   * Get feed by ID
   */
  async getFeedById(id: string): Promise<Feed | null> {
    const feed = await this.feedRepository.findOne({ where: { id } });
    return feed ? this.mapEntityToDTO(feed) : null;
  }
  
  /**
   * Get all feeds
   */
  async getAllFeeds(): Promise<Feed[]> {
    const feeds = await this.feedRepository.find({
      order: { created_at: 'DESC' },
    });
    return feeds.map(feed => this.mapEntityToDTO(feed));
  }
  
  /**
   * Get active feeds
   */
  async getActiveFeeds(): Promise<Feed[]> {
    const feeds = await this.feedRepository.find({
      where: { is_active: true },
      order: { created_at: 'DESC' },
    });
    return feeds.map(feed => this.mapEntityToDTO(feed));
  }
  
  /**
   * Update feed
   */
  async updateFeed(id: string, updates: Partial<Feed>): Promise<Feed> {
    if (!id) {
      throw new Error('Feed ID is required');
    }

    try {
      const feed = await this.feedRepository.findOne({ where: { id } });
      if (!feed) {
        throw new Error(`Feed with id ${id} not found`);
      }
      
      if (updates.name !== undefined) {
        if (!updates.name.trim()) {
          throw new Error('Feed name cannot be empty');
        }
        feed.name = updates.name;
      }
      if (updates.feedType !== undefined) feed.feed_type = updates.feedType;
      if (updates.apiEndpoint !== undefined) feed.api_endpoint = updates.apiEndpoint;
      if (updates.apiKeyEncrypted !== undefined) feed.api_key_encrypted = updates.apiKeyEncrypted;
      if (updates.syncIntervalMinutes !== undefined) {
        if (updates.syncIntervalMinutes < 1) {
          throw new Error('Sync interval must be at least 1 minute');
        }
        feed.sync_interval_minutes = updates.syncIntervalMinutes;
      }
      if (updates.isActive !== undefined) feed.is_active = updates.isActive;
      if (updates.reliabilityScore !== undefined) {
        const score = Math.max(0, Math.min(100, updates.reliabilityScore));
        feed.reliability_score = score;
      }
      
      const saved = await this.feedRepository.save(feed);
      return this.mapEntityToDTO(saved);
    } catch (error: any) {
      if (error.message.includes('not found') || error.message.includes('required')) {
        throw error;
      }
      logger.error('Failed to update feed', { error, id, updates });
      throw error;
    }
  }
  
  /**
   * Delete feed
   */
  async deleteFeed(id: string): Promise<void> {
    const feed = await this.feedRepository.findOne({ where: { id } });
    if (!feed) {
      throw new Error(`Feed with id ${id} not found`);
    }
    
    await this.feedRepository.remove(feed);
  }
  
  /**
   * Toggle feed active status
   */
  async toggleFeed(id: string): Promise<Feed> {
    const feed = await this.feedRepository.findOne({ where: { id } });
    if (!feed) {
      throw new Error(`Feed with id ${id} not found`);
    }
    
    feed.is_active = !feed.is_active;
    const saved = await this.feedRepository.save(feed);
    return this.mapEntityToDTO(saved);
  }
  
  /**
   * Update feed sync status
   */
  async updateSyncStatus(
    id: string,
    status: 'success' | 'failed' | 'partial',
    error?: string
  ): Promise<void> {
    const feed = await this.feedRepository.findOne({ where: { id } });
    if (!feed) {
      logger.warn(`Feed ${id} not found for sync status update`);
      return;
    }
    
    feed.last_sync_at = new Date();
    feed.last_sync_status = status;
    feed.last_sync_error = error || null;
    
    // Update reliability score based on sync status
    if (status === 'success') {
      feed.reliability_score = Math.min(100, (feed.reliability_score || 50) + 1);
    } else if (status === 'failed') {
      feed.reliability_score = Math.max(0, (feed.reliability_score || 50) - 5);
    }
    
    await this.feedRepository.save(feed);
  }
  
  /**
   * Get feed by name
   */
  async getFeedByName(name: string): Promise<Feed | null> {
    const feed = await this.feedRepository.findOne({ where: { name } });
    return feed ? this.mapEntityToDTO(feed) : null;
  }
  
  /**
   * Map TypeORM entity to DTO
   */
  private mapEntityToDTO(entity: FeedEntity): Feed {
    return {
      id: entity.id,
      name: entity.name,
      feedType: entity.feed_type as 'misp' | 'otx' | 'custom' | 'user_submitted',
      apiEndpoint: entity.api_endpoint || null,
      apiKeyEncrypted: entity.api_key_encrypted || null,
      syncIntervalMinutes: entity.sync_interval_minutes,
      lastSyncAt: entity.last_sync_at || null,
      lastSyncStatus: (entity.last_sync_status as 'success' | 'failed' | 'partial') || null,
      lastSyncError: entity.last_sync_error || null,
      isActive: entity.is_active,
      reliabilityScore: entity.reliability_score ? parseFloat(entity.reliability_score.toString()) : undefined,
      createdAt: entity.created_at,
      updatedAt: entity.updated_at,
    };
  }
}
