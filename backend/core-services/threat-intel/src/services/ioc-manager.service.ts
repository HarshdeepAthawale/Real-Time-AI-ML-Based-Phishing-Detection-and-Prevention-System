import { DataSource, Repository, QueryRunner } from 'typeorm';
import { IOC as IOCEntity } from '../../../../shared/database/models/IOC';
import { IOC, IOCSearchParams, IOCStats, IOCType, Severity } from '../models/ioc.model';
import { logger } from '../utils/logger';
import { normalizeIOCValue, hashIOCValue } from '../utils/normalizers';

export class IOCManagerService {
  private iocRepository: Repository<IOCEntity>;
  
  constructor(private dataSource: DataSource, private queryRunner?: QueryRunner) {
    // Use query runner's manager if available (for transaction isolation in tests)
    // Otherwise use dataSource manager (normal operation)
    const manager = queryRunner?.manager || dataSource.manager || dataSource;
    this.iocRepository = manager.getRepository(IOCEntity);
  }
  
  /**
   * Create or update an IOC
   */
  async createIOC(ioc: Omit<IOC, 'id' | 'createdAt' | 'updatedAt'>): Promise<IOC> {
    const normalizedValue = normalizeIOCValue(ioc.iocType, ioc.iocValue);
    const iocValueHash = hashIOCValue(normalizedValue);
    
    try {
      // Try to find existing IOC
      const existing = await this.iocRepository.findOne({
        where: {
          ioc_type: ioc.iocType,
          ioc_value_hash: iocValueHash,
        },
      });

      if (existing) {
        // Update existing IOC
        existing.last_seen_at = new Date();
        existing.source_reports = (existing.source_reports || 1) + 1;
        
        // Merge metadata
        if (ioc.metadata) {
          existing.metadata = {
            ...(existing.metadata || {}),
            ...ioc.metadata,
          };
        }
        
        // Update other fields if provided
        if (ioc.severity) existing.severity = ioc.severity;
        if (ioc.confidence !== undefined) existing.confidence = ioc.confidence;
        if (ioc.threatType) existing.threat_type = ioc.threatType;
        
        const updated = await this.iocRepository.save(existing);
        return this.mapEntityToDTO(updated);
      }

      // Create new IOC
      const newIOC = this.iocRepository.create({
        feed_id: ioc.feedId || null,
        ioc_type: ioc.iocType,
        ioc_value: normalizedValue,
        ioc_value_hash: iocValueHash,
        threat_type: ioc.threatType || null,
        severity: ioc.severity || null,
        confidence: ioc.confidence || 50,
        first_seen_at: ioc.firstSeenAt || new Date(),
        last_seen_at: ioc.lastSeenAt || new Date(),
        source_reports: 1,
        metadata: ioc.metadata || {},
      });

      const saved = await this.iocRepository.save(newIOC);
      return this.mapEntityToDTO(saved);
    } catch (error) {
      logger.error('Failed to create IOC', { error, ioc });
      throw error;
    }
  }
  
  /**
   * Find IOC by type and value
   */
  async findIOC(iocType: IOCType, iocValue: string): Promise<IOC | null> {
    if (!iocType || !iocValue) {
      return null;
    }

    try {
      const normalizedValue = normalizeIOCValue(iocType, iocValue);
      const iocValueHash = hashIOCValue(normalizedValue);
      
      const found = await this.iocRepository.findOne({
        where: {
          ioc_type: iocType,
          ioc_value_hash: iocValueHash,
        },
      });
      
      return found ? this.mapEntityToDTO(found) : null;
    } catch (error) {
      logger.error('Failed to find IOC', { error, iocType, iocValue });
      throw error;
    }
  }
  
  /**
   * Bulk create IOCs
   */
  async bulkCreateIOCs(iocs: Array<Omit<IOC, 'id' | 'createdAt' | 'updatedAt'>>): Promise<number> {
    if (!iocs || iocs.length === 0) {
      return 0;
    }

    let inserted = 0;
    const errors: Array<{ ioc: string; error: string }> = [];
    
    for (const ioc of iocs) {
      try {
        // Validate IOC before processing
        if (!ioc.iocType || !ioc.iocValue) {
          logger.warn('Skipping invalid IOC: missing type or value', { ioc });
          continue;
        }

        await this.createIOC(ioc);
        inserted++;
      } catch (error: any) {
        const errorMessage = error?.message || String(error);
        errors.push({ ioc: ioc.iocValue || 'unknown', error: errorMessage });
        logger.error(`Failed to insert IOC: ${ioc.iocValue}`, error);
      }
    }
    
    if (errors.length > 0 && errors.length === iocs.length) {
      // All IOCs failed - this might indicate a bigger problem
      logger.error(`All ${iocs.length} IOCs failed to insert`, { errors: errors.slice(0, 5) });
    } else if (errors.length > 0) {
      logger.warn(`${errors.length} out of ${iocs.length} IOCs failed to insert`);
    }
    
    return inserted;
  }
  
  /**
   * Search IOCs with filters
   */
  async searchIOCs(params: IOCSearchParams): Promise<{ iocs: IOC[]; total: number }> {
    try {
      const queryBuilder = this.iocRepository.createQueryBuilder('ioc');
      
      if (params.iocType) {
        queryBuilder.andWhere('ioc.ioc_type = :iocType', { iocType: params.iocType });
      }
      
      if (params.severity) {
        queryBuilder.andWhere('ioc.severity = :severity', { severity: params.severity });
      }
      
      if (params.threatType) {
        queryBuilder.andWhere('ioc.threat_type ILIKE :threatType', { 
          threatType: `%${params.threatType}%` 
        });
      }
      
      if (params.search) {
        queryBuilder.andWhere(
          '(ioc.ioc_value ILIKE :search OR ioc.threat_type ILIKE :search)',
          { search: `%${params.search}%` }
        );
      }
      
      const total = await queryBuilder.getCount();
      
      // Validate and sanitize limit/offset
      const limit = Math.min(Math.max(1, params.limit || 50), 1000); // Max 1000
      const offset = Math.max(0, params.offset || 0);
      
      queryBuilder
        .orderBy('ioc.last_seen_at', 'DESC')
        .limit(limit)
        .offset(offset);
      
      const entities = await queryBuilder.getMany();
      const iocs = entities.map(entity => this.mapEntityToDTO(entity));
      
      return { iocs, total };
    } catch (error) {
      logger.error('Failed to search IOCs', { error, params });
      throw error;
    }
  }
  
  /**
   * Get IOC statistics
   */
  async getStats(): Promise<IOCStats> {
    try {
      const total = await this.iocRepository.count();
    
    // Count by type
    const byType = await this.iocRepository
      .createQueryBuilder('ioc')
      .select('ioc.ioc_type', 'type')
      .addSelect('COUNT(*)', 'count')
      .groupBy('ioc.ioc_type')
      .getRawMany();
    
    const typeMap: Record<IOCType, number> = {
      url: 0,
      domain: 0,
      ip: 0,
      email: 0,
      hash_md5: 0,
      hash_sha1: 0,
      hash_sha256: 0,
      filename: 0,
    };
    
    byType.forEach((row: any) => {
      if (row.type in typeMap) {
        typeMap[row.type as IOCType] = parseInt(row.count, 10);
      }
    });
    
    // Count by severity
    const bySeverity = await this.iocRepository
      .createQueryBuilder('ioc')
      .select('ioc.severity', 'severity')
      .addSelect('COUNT(*)', 'count')
      .where('ioc.severity IS NOT NULL')
      .groupBy('ioc.severity')
      .getRawMany();
    
    const severityMap: Record<Severity, number> = {
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
    };
    
    bySeverity.forEach((row: any) => {
      if (row.severity in severityMap) {
        severityMap[row.severity as Severity] = parseInt(row.count, 10);
      }
    });
    
    // Count by feed
    const byFeed = await this.iocRepository
      .createQueryBuilder('ioc')
      .select('ioc.feed_id', 'feedId')
      .addSelect('COUNT(*)', 'count')
      .where('ioc.feed_id IS NOT NULL')
      .groupBy('ioc.feed_id')
      .getRawMany();
    
    const feedMap: Record<string, number> = {};
    byFeed.forEach((row: any) => {
      feedMap[row.feedId] = parseInt(row.count, 10);
    });
    
    // Recent count (last 24 hours)
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    
    const recentCountResult = await this.iocRepository
      .createQueryBuilder('ioc')
      .where('ioc.created_at >= :yesterday', { yesterday })
      .getCount();
    
      return {
        total,
        byType: typeMap,
        bySeverity: severityMap,
        byFeed: feedMap,
        recentCount: recentCountResult,
      };
    } catch (error) {
      logger.error('Failed to get IOC statistics', error);
      throw error;
    }
  }
  
  /**
   * Map TypeORM entity to DTO
   */
  private mapEntityToDTO(entity: IOCEntity): IOC {
    return {
      id: entity.id,
      feedId: entity.feed_id || null,
      iocType: entity.ioc_type as IOCType,
      iocValue: entity.ioc_value,
      threatType: entity.threat_type || null,
      severity: (entity.severity as Severity) || null,
      confidence: entity.confidence ? parseFloat(entity.confidence.toString()) : null,
      firstSeenAt: entity.first_seen_at || null,
      lastSeenAt: entity.last_seen_at || null,
      source: entity.feed_id ? 'feed' : 'user',
      sourceReports: entity.source_reports || 0,
      metadata: entity.metadata || {},
      createdAt: entity.created_at,
      updatedAt: entity.updated_at,
    };
  }
}
