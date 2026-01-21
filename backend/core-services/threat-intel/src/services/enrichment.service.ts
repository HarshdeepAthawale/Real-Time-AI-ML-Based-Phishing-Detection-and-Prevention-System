import { IOC, IOCType } from '../models/ioc.model';
import { IOCManagerService } from './ioc-manager.service';
import { logger } from '../utils/logger';
import { normalizeIOCValue } from '../utils/normalizers';

export interface EnrichmentResult {
  enriched: boolean;
  additionalContext?: {
    relatedIOCs?: IOC[];
    threatType?: string;
    confidence?: number;
    recommendations?: string[];
  };
}

/**
 * Service for enriching IOCs with additional context and metadata
 */
export class EnrichmentService {
  private iocManager: IOCManagerService;
  
  constructor(iocManager: IOCManagerService) {
    this.iocManager = iocManager;
  }
  
  /**
   * Enrich an IOC with additional context
   */
  async enrichIOC(ioc: IOC): Promise<EnrichmentResult> {
    try {
      const enrichment: EnrichmentResult = {
        enriched: false,
        additionalContext: {},
      };
      
      // Find related IOCs (same domain for URLs, same IP, etc.)
      const relatedIOCs = await this.findRelatedIOCs(ioc);
      if (relatedIOCs.length > 0) {
        enrichment.enriched = true;
        enrichment.additionalContext!.relatedIOCs = relatedIOCs;
        
        // Calculate aggregate confidence based on related IOCs
        const avgConfidence = relatedIOCs.reduce((sum, related) => 
          sum + (related.confidence || 50), 0
        ) / relatedIOCs.length;
        
        if (avgConfidence > (ioc.confidence || 50)) {
          enrichment.additionalContext!.confidence = Math.min(100, avgConfidence + 10);
        }
      }
      
      // Determine threat type based on related IOCs
      const threatTypes = new Set(relatedIOCs.map(r => r.threatType).filter(Boolean));
      if (threatTypes.size > 0) {
        const mostCommonThreatType = this.getMostCommon(
          relatedIOCs.map(r => r.threatType).filter(Boolean) as string[]
        );
        if (mostCommonThreatType) {
          enrichment.additionalContext!.threatType = mostCommonThreatType;
          enrichment.enriched = true;
        }
      }
      
      // Generate recommendations based on IOC type and related data
      const recommendations = this.generateRecommendations(ioc, relatedIOCs);
      if (recommendations.length > 0) {
        enrichment.enriched = true;
        enrichment.additionalContext!.recommendations = recommendations;
      }
      
      return enrichment;
    } catch (error) {
      logger.error('IOC enrichment error', error);
      return { enriched: false };
    }
  }
  
  /**
   * Find related IOCs based on IOC type and value
   */
  private async findRelatedIOCs(ioc: IOC): Promise<IOC[]> {
    const related: IOC[] = [];
    
    try {
      if (ioc.iocType === 'url') {
        // Extract domain from URL and find related domain IOCs
        try {
          const url = new URL(ioc.iocValue);
          const domain = url.hostname;
          
          const domainIOCs = await this.iocManager.searchIOCs({
            iocType: 'domain',
            search: domain,
            limit: 10,
          });
          related.push(...domainIOCs.iocs);
        } catch {
          // Invalid URL, skip
        }
      } else if (ioc.iocType === 'domain') {
        // Find URLs with the same domain
        const urlIOCs = await this.iocManager.searchIOCs({
          iocType: 'url',
          search: ioc.iocValue,
          limit: 10,
        });
        related.push(...urlIOCs.iocs);
      } else if (ioc.iocType === 'ip') {
        // Find other IOCs with the same IP
        const ipIOCs = await this.iocManager.searchIOCs({
          iocType: 'ip',
          search: ioc.iocValue,
          limit: 10,
        });
        related.push(...ipIOCs.iocs.filter(r => r.id !== ioc.id));
      }
      
      // Find IOCs with the same threat type
      if (ioc.threatType) {
        const threatTypeIOCs = await this.iocManager.searchIOCs({
          threatType: ioc.threatType,
          limit: 5,
        });
        related.push(...threatTypeIOCs.iocs.filter(r => r.id !== ioc.id));
      }
    } catch (error) {
      logger.error('Error finding related IOCs', error);
    }
    
    // Remove duplicates based on ID
    const uniqueRelated = related.filter((value, index, self) =>
      index === self.findIndex((t) => t.id === value.id)
    );
    
    return uniqueRelated.slice(0, 10); // Limit to 10 related IOCs
  }
  
  /**
   * Generate recommendations based on IOC data
   */
  private generateRecommendations(ioc: IOC, relatedIOCs: IOC[]): string[] {
    const recommendations: string[] = [];
    
    if (ioc.severity === 'critical' || ioc.severity === 'high') {
      recommendations.push('Immediate action recommended - high severity IOC detected');
    }
    
    if (relatedIOCs.length > 5) {
      recommendations.push(`Multiple related IOCs detected (${relatedIOCs.length}) - possible coordinated attack`);
    }
    
    if (ioc.iocType === 'domain' || ioc.iocType === 'url') {
      recommendations.push('Consider blocking this domain/URL at network level');
      recommendations.push('Check DNS logs for related queries');
    }
    
    if (ioc.iocType === 'ip') {
      recommendations.push('Consider blocking this IP address at firewall');
      recommendations.push('Review network traffic logs for this IP');
    }
    
    if (ioc.iocType === 'hash_md5' || ioc.iocType === 'hash_sha1' || ioc.iocType === 'hash_sha256') {
      recommendations.push('Scan systems for files matching this hash');
      recommendations.push('Check antivirus logs for detections');
    }
    
    if (ioc.confidence && ioc.confidence >= 90) {
      recommendations.push('High confidence IOC - recommended for immediate blocking');
    }
    
    return recommendations;
  }
  
  /**
   * Get most common value from array
   */
  private getMostCommon<T>(arr: T[]): T | undefined {
    if (arr.length === 0) return undefined;
    
    const frequency: Map<T, number> = new Map();
    for (const item of arr) {
      frequency.set(item, (frequency.get(item) || 0) + 1);
    }
    
    let maxFreq = 0;
    let mostCommon: T | undefined;
    for (const [item, freq] of frequency.entries()) {
      if (freq > maxFreq) {
        maxFreq = freq;
        mostCommon = item;
      }
    }
    
    return mostCommon;
  }
  
  /**
   * Bulk enrich multiple IOCs
   */
  async bulkEnrichIOCs(iocs: IOC[]): Promise<Map<string, EnrichmentResult>> {
    const results = new Map<string, EnrichmentResult>();
    
    for (const ioc of iocs) {
      if (ioc.id || ioc.iocValue) {
        const key = ioc.id || ioc.iocValue;
        const enrichment = await this.enrichIOC(ioc);
        results.set(key, enrichment);
      }
    }
    
    return results;
  }
}
