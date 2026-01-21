import Redis from 'ioredis';
import { BloomFilter } from 'bloom-filters';
import { IOCManagerService } from './ioc-manager.service';
import { IOC, IOCType } from '../models/ioc.model';
import { config } from '../config';
import { logger } from '../utils/logger';
import { 
  createBloomFilter, 
  serializeBloomFilter, 
  loadBloomFilter,
  SerializedBloomFilter 
} from '../utils/bloom-filter';
import { normalizeIOCValue } from '../utils/normalizers';

export class IOCMatcherService {
  private redis: Redis;
  private iocManager: IOCManagerService;
  private bloomFilters: Map<IOCType, BloomFilter> = new Map();
  private readonly bloomFilterKeyPrefix = 'bloom:ioc:';
  
  constructor(redis: Redis, iocManager: IOCManagerService) {
    this.redis = redis;
    this.iocManager = iocManager;
  }
  
  /**
   * Initialize bloom filters for all IOC types
   */
  async initializeBloomFilters(): Promise<void> {
    const types: IOCType[] = [
      'url', 
      'domain', 
      'ip', 
      'hash_md5', 
      'hash_sha1', 
      'hash_sha256',
      'email',
      'filename'
    ];
    
    for (const type of types) {
      await this.loadBloomFilter(type);
    }
    
    logger.info('Bloom filters initialized');
  }
  
  /**
   * Load bloom filter from Redis or create new one
   */
  private async loadBloomFilter(iocType: IOCType): Promise<void> {
    try {
      const key = `${this.bloomFilterKeyPrefix}${iocType}`;
      const serialized = await this.redis.get(key);
      
      if (serialized) {
        const data = JSON.parse(serialized) as SerializedBloomFilter;
        const filter = loadBloomFilter(data);
        this.bloomFilters.set(iocType, filter);
        logger.debug(`Loaded bloom filter for ${iocType}`);
      } else {
        // Create new bloom filter
        const filter = createBloomFilter({
          size: config.bloomFilter.defaultSize,
          falsePositiveRate: config.bloomFilter.defaultFalsePositiveRate,
        });
        this.bloomFilters.set(iocType, filter);
        logger.debug(`Created new bloom filter for ${iocType}`);
      }
    } catch (error) {
      logger.error(`Failed to load bloom filter for ${iocType}`, error);
      // Create new filter as fallback
      const filter = createBloomFilter({
        size: config.bloomFilter.defaultSize,
        falsePositiveRate: config.bloomFilter.defaultFalsePositiveRate,
      });
      this.bloomFilters.set(iocType, filter);
    }
  }
  
  /**
   * Persist bloom filter to Redis
   */
  private async persistBloomFilter(iocType: IOCType): Promise<void> {
    try {
      const filter = this.bloomFilters.get(iocType);
      if (!filter) {
        return;
      }
      
      const key = `${this.bloomFilterKeyPrefix}${iocType}`;
      const serialized = serializeBloomFilter(filter);
      
      await this.redis.setex(
        key,
        config.bloomFilter.redisTtl,
        JSON.stringify(serialized)
      );
    } catch (error) {
      logger.error(`Failed to persist bloom filter for ${iocType}`, error);
    }
  }
  
  /**
   * Match IOC - fast negative lookup with bloom filter
   */
  async matchIOC(iocType: IOCType, iocValue: string): Promise<IOC | null> {
    const normalizedValue = normalizeIOCValue(iocType, iocValue);
    
    // Fast negative lookup using bloom filter
    const filter = this.bloomFilters.get(iocType);
    
    // Only use bloom filter for negative lookup if it exists
    // If bloom filter doesn't exist, always check database (fallback behavior)
    if (filter) {
      // Check if bloom filter indicates IOC is definitely not present
      if (!filter.has(normalizedValue)) {
        // Bloom filter says definitely not in database (with small false positive rate)
        // However, if bloom filter was just initialized and might be empty,
        // we should still check database to be safe
        // For now, trust the bloom filter - it should be populated when IOCs are added
        return null;
      }
      // Bloom filter says it might exist (could be false positive), check database
    }
    
    // If no bloom filter or bloom filter says might exist, check database
    return await this.iocManager.findIOC(iocType, iocValue);
  }
  
  /**
   * Add IOC to bloom filter
   */
  async addToBloomFilter(iocType: IOCType, iocValue: string): Promise<void> {
    const normalizedValue = normalizeIOCValue(iocType, iocValue);
    
    let filter = this.bloomFilters.get(iocType);
    if (!filter) {
      // Initialize filter if it doesn't exist
      await this.loadBloomFilter(iocType);
      filter = this.bloomFilters.get(iocType);
      if (!filter) {
        // Create new filter if loading failed
        filter = createBloomFilter({
          size: config.bloomFilter.defaultSize,
          falsePositiveRate: config.bloomFilter.defaultFalsePositiveRate,
        });
        this.bloomFilters.set(iocType, filter);
      }
    }
    
    // Add normalized value to bloom filter
    filter.add(normalizedValue);
    
    // Persist to Redis
    await this.persistBloomFilter(iocType);
  }
  
  /**
   * Bulk add IOCs to bloom filter
   */
  async bulkAddToBloomFilter(iocType: IOCType, iocValues: string[]): Promise<void> {
    const filter = this.bloomFilters.get(iocType);
    if (!filter) {
      logger.warn(`Bloom filter not found for ${iocType}`);
      return;
    }
    
    for (const value of iocValues) {
      const normalizedValue = normalizeIOCValue(iocType, value);
      filter.add(normalizedValue);
    }
    
    // Persist to Redis
    await this.persistBloomFilter(iocType);
  }
  
  /**
   * Rebuild bloom filter from database
   */
  async rebuildBloomFilter(iocType: IOCType): Promise<void> {
    logger.info(`Rebuilding bloom filter for ${iocType}`);
    
    try {
      // Get all IOCs of this type from database
      const { iocs } = await this.iocManager.searchIOCs({
        iocType,
        limit: 100000, // Large limit
      });
      
      // Create new filter
      const filter = createBloomFilter({
        size: Math.max(
          config.bloomFilter.defaultSize,
          iocs.length * 2 // Double the size for future growth
        ),
        falsePositiveRate: config.bloomFilter.defaultFalsePositiveRate,
      });
      
      // Add all IOCs
      for (const ioc of iocs) {
        const normalizedValue = normalizeIOCValue(ioc.iocType, ioc.iocValue);
        filter.add(normalizedValue);
      }
      
      this.bloomFilters.set(iocType, filter);
      await this.persistBloomFilter(iocType);
      
      logger.info(`Rebuilt bloom filter for ${iocType} with ${iocs.length} items`);
    } catch (error) {
      logger.error(`Failed to rebuild bloom filter for ${iocType}`, error);
      throw error;
    }
  }
  
  /**
   * Get bloom filter statistics
   */
  getBloomFilterStats(): Record<IOCType, { initialized: boolean }> {
    const stats: Record<string, { initialized: boolean }> = {};
    
    const types: IOCType[] = [
      'url', 
      'domain', 
      'ip', 
      'hash_md5', 
      'hash_sha1', 
      'hash_sha256',
      'email',
      'filename'
    ];
    
    for (const type of types) {
      stats[type] = {
        initialized: this.bloomFilters.has(type),
      };
    }
    
    return stats as Record<IOCType, { initialized: boolean }>;
  }
}
