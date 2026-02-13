import { BloomFilterService } from './bloom-filter.service';
import { getPostgreSQL } from '../../../../shared/database/connection';
import { IOC as IOCEntity } from '../../../../shared/database/models/IOC';
import { In } from 'typeorm';
import { logger } from '../utils/logger';

export interface IOCMatchResult {
  found: boolean;
  matches: Array<{
    id: string;
    ioc_type: string;
    ioc_value: string;
    source: string;
    threat_type: string;
    confidence: number;
    severity: string;
    first_seen_at: Date;
    last_seen_at: Date;
  }>;
  bloom_filter_hit: boolean;
}

export class IOCMatcherService {
  private urlBloomFilter: BloomFilterService;
  private domainBloomFilter: BloomFilterService;
  private ipBloomFilter: BloomFilterService;
  private hashBloomFilter: BloomFilterService;
  private emailBloomFilter: BloomFilterService;
  
  private lastRebuildTime: Date;
  private readonly rebuildIntervalHours = 1;

  constructor(_redis?: any, _iocManager?: any) {
    // Initialize Bloom filters for different IOC types
    const config = {
      expectedItems: parseInt(process.env.BLOOM_FILTER_SIZE || '1000000'),
      falsePositiveRate: parseFloat(process.env.BLOOM_FILTER_FALSE_POSITIVE_RATE || '0.01')
    };

    this.urlBloomFilter = new BloomFilterService(config.expectedItems, config.falsePositiveRate);
    this.domainBloomFilter = new BloomFilterService(config.expectedItems, config.falsePositiveRate);
    this.ipBloomFilter = new BloomFilterService(config.expectedItems / 10, config.falsePositiveRate);
    this.hashBloomFilter = new BloomFilterService(config.expectedItems / 2, config.falsePositiveRate);
    this.emailBloomFilter = new BloomFilterService(config.expectedItems / 5, config.falsePositiveRate);
    
    this.lastRebuildTime = new Date();

    logger.info('IOC Matcher initialized with Bloom filters');
  }

  /**
   * Build Bloom filters from database IOCs
   */
  async buildBloomFilters(): Promise<void> {
    try {
      const dataSource = getPostgreSQL();
      const iocRepository = dataSource.getRepository(IOCEntity);

      // Fetch active IOCs
      const iocs = await iocRepository.find({
        where: { is_active: true },
        select: ['ioc_type', 'ioc_value']
      });

      // Clear existing filters
      this.urlBloomFilter.clear();
      this.domainBloomFilter.clear();
      this.ipBloomFilter.clear();
      this.hashBloomFilter.clear();
      this.emailBloomFilter.clear();

      // Populate filters
      let urlCount = 0, domainCount = 0, ipCount = 0, hashCount = 0, emailCount = 0;

      for (const ioc of iocs) {
        switch (ioc.ioc_type) {
          case 'url':
            this.urlBloomFilter.add(ioc.ioc_value);
            urlCount++;
            break;
          case 'domain':
          case 'hostname':
            this.domainBloomFilter.add(ioc.ioc_value);
            domainCount++;
            break;
          case 'ip':
          case 'ipv4':
          case 'ipv6':
            this.ipBloomFilter.add(ioc.ioc_value);
            ipCount++;
            break;
          case 'hash':
          case 'md5':
          case 'sha1':
          case 'sha256':
          case 'sha512':
            this.hashBloomFilter.add(ioc.ioc_value);
            hashCount++;
            break;
          case 'email':
            this.emailBloomFilter.add(ioc.ioc_value);
            emailCount++;
            break;
        }
      }

      this.lastRebuildTime = new Date();

      logger.info(`Bloom filters rebuilt: ${urlCount} URLs, ${domainCount} domains, ` +
                  `${ipCount} IPs, ${hashCount} hashes, ${emailCount} emails`);
      logger.info(`Total IOCs: ${iocs.length}`);
      
      // Log filter statistics
      logger.info('URL filter stats:', this.urlBloomFilter.getStats());
      logger.info('Domain filter stats:', this.domainBloomFilter.getStats());
      
    } catch (error: any) {
      logger.error(`Failed to build Bloom filters: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Alias for buildBloomFilters (backward compatibility)
   */
  async initializeBloomFilters(): Promise<void> {
    return this.buildBloomFilters();
  }

  /**
   * Match IOC by type and value (routes to specific matcher)
   */
  async matchIOC(iocType: string, value: string): Promise<IOCMatchResult> {
    const type = iocType.toLowerCase();
    if (type === 'url') return this.matchURL(value);
    if (['domain', 'hostname'].includes(type)) return this.matchDomain(value);
    if (['ip', 'ipv4', 'ipv6'].includes(type)) return this.matchIP(value);
    if (['hash', 'md5', 'sha1', 'sha256', 'sha512'].includes(type)) return this.matchHash(value);
    if (type === 'email') return this.matchEmail(value);
    return { found: false, matches: [], bloom_filter_hit: false };
  }

  /**
   * Add single IOC to Bloom filter (convenience wrapper)
   */
  async addToBloomFilter(iocType: string, value: string): Promise<void> {
    return this.bulkAddToBloomFilter(iocType, [value]);
  }

  /**
   * Add multiple IOCs to Bloom filter by type
   */
  async bulkAddToBloomFilter(iocType: string, values: string[]): Promise<void> {
    const filter = this.getBloomFilterForType(iocType);
    if (!filter) {
      logger.warn(`No Bloom filter for IOC type: ${iocType}`);
      return;
    }
    for (const value of values) {
      filter.add(value);
    }
  }

  private getBloomFilterForType(iocType: string): BloomFilterService | null {
    switch (iocType.toLowerCase()) {
      case 'url': return this.urlBloomFilter;
      case 'domain':
      case 'hostname': return this.domainBloomFilter;
      case 'ip':
      case 'ipv4':
      case 'ipv6': return this.ipBloomFilter;
      case 'hash':
      case 'md5':
      case 'sha1':
      case 'sha256':
      case 'sha512': return this.hashBloomFilter;
      case 'email': return this.emailBloomFilter;
      default: return null;
    }
  }

  /**
   * Check if Bloom filters need rebuilding
   */
  needsRebuild(): boolean {
    const hoursSinceRebuild = (Date.now() - this.lastRebuildTime.getTime()) / 3600000;
    return hoursSinceRebuild >= this.rebuildIntervalHours;
  }

  /**
   * Match URL against IOCs
   */
  async matchURL(url: string): Promise<IOCMatchResult> {
    // Quick Bloom filter check
    const bloomHit = this.urlBloomFilter.mightContain(url);
    
    if (!bloomHit) {
      // Definitely not in database
      return { found: false, matches: [], bloom_filter_hit: false };
    }

    // Verify in database (Bloom filter can have false positives)
    return this.verifyIOC('url', url, bloomHit);
  }

  /**
   * Match domain against IOCs
   */
  async matchDomain(domain: string): Promise<IOCMatchResult> {
    const bloomHit = this.domainBloomFilter.mightContain(domain);
    
    if (!bloomHit) {
      return { found: false, matches: [], bloom_filter_hit: false };
    }

    return this.verifyIOC('domain', domain, bloomHit);
  }

  /**
   * Match IP against IOCs
   */
  async matchIP(ip: string): Promise<IOCMatchResult> {
    const bloomHit = this.ipBloomFilter.mightContain(ip);
    
    if (!bloomHit) {
      return { found: false, matches: [], bloom_filter_hit: false };
    }

    return this.verifyIOC('ip', ip, bloomHit);
  }

  /**
   * Match hash against IOCs
   */
  async matchHash(hash: string): Promise<IOCMatchResult> {
    const bloomHit = this.hashBloomFilter.mightContain(hash);
    
    if (!bloomHit) {
      return { found: false, matches: [], bloom_filter_hit: false };
    }

    return this.verifyIOC('hash', hash, bloomHit);
  }

  /**
   * Match email against IOCs
   */
  async matchEmail(email: string): Promise<IOCMatchResult> {
    const bloomHit = this.emailBloomFilter.mightContain(email);
    
    if (!bloomHit) {
      return { found: false, matches: [], bloom_filter_hit: false };
    }

    return this.verifyIOC('email', email, bloomHit);
  }

  /**
   * Verify IOC in database after Bloom filter hit
   */
  private async verifyIOC(type: string, value: string, bloomHit: boolean): Promise<IOCMatchResult> {
    try {
      const dataSource = getPostgreSQL();
      const iocRepository = dataSource.getRepository(IOCEntity);

      // Query database for exact match
      const typeVariants = [type, `${type}s`];
      const iocs = await iocRepository.find({
        where: {
          ioc_type: In(typeVariants) as any,
          ioc_value: value,
          is_active: true
        },
        order: {
          confidence: 'DESC',
          last_seen_at: 'DESC'
        }
      });

      if (iocs.length === 0) {
        // False positive from Bloom filter
        return { found: false, matches: [], bloom_filter_hit: bloomHit };
      }

      // Map to response format
      const matches = iocs.map(ioc => ({
        id: ioc.id,
        ioc_type: ioc.ioc_type,
        ioc_value: ioc.ioc_value,
        source: (ioc as any).source ?? ioc.feed_id ?? 'unknown',
        threat_type: ioc.threat_type || 'unknown',
        confidence: ioc.confidence || 0,
        severity: ioc.severity || 'medium',
        first_seen_at: ioc.first_seen_at,
        last_seen_at: ioc.last_seen_at
      }));

      return {
        found: true,
        matches,
        bloom_filter_hit: bloomHit
      };

    } catch (error: any) {
      logger.error(`Failed to verify IOC in database: ${error.message}`);
      return { found: false, matches: [], bloom_filter_hit: bloomHit };
    }
  }

  /**
   * Match multiple IOCs in batch
   */
  async matchBatch(iocs: Array<{ type: string; value: string }>): Promise<Map<string, IOCMatchResult>> {
    const results = new Map<string, IOCMatchResult>();

    for (const ioc of iocs) {
      const key = `${ioc.type}:${ioc.value}`;
      
      switch (ioc.type) {
        case 'url':
          results.set(key, await this.matchURL(ioc.value));
          break;
        case 'domain':
          results.set(key, await this.matchDomain(ioc.value));
          break;
        case 'ip':
          results.set(key, await this.matchIP(ioc.value));
          break;
        case 'hash':
          results.set(key, await this.matchHash(ioc.value));
          break;
        case 'email':
          results.set(key, await this.matchEmail(ioc.value));
          break;
        default:
          logger.warn(`Unknown IOC type: ${ioc.type}`);
      }
    }

    return results;
  }

  /**
   * Get matcher statistics
   */
  getStats(): {
    lastRebuildTime: Date;
    needsRebuild: boolean;
    filters: {
      url: any;
      domain: any;
      ip: any;
      hash: any;
      email: any;
    };
  } {
    return {
      lastRebuildTime: this.lastRebuildTime,
      needsRebuild: this.needsRebuild(),
      filters: {
        url: this.urlBloomFilter.getStats(),
        domain: this.domainBloomFilter.getStats(),
        ip: this.ipBloomFilter.getStats(),
        hash: this.hashBloomFilter.getStats(),
        email: this.emailBloomFilter.getStats()
      }
    };
  }

  /**
   * Extract URLs from text
   */
  extractURLs(text: string): string[] {
    const urlPattern = /https?:\/\/[^\s<>"{}|\\^`\[\]]+/g;
    return text.match(urlPattern) || [];
  }

  /**
   * Extract domains from URLs
   */
  extractDomains(urls: string[]): string[] {
    const domains = new Set<string>();
    
    for (const url of urls) {
      try {
        const urlObj = new URL(url);
        domains.add(urlObj.hostname);
      } catch {
        // Invalid URL, skip
      }
    }
    
    return [...domains];
  }

  /**
   * Extract IPs from text
   */
  extractIPs(text: string): string[] {
    const ipPattern = /\b(?:\d{1,3}\.){3}\d{1,3}\b/g;
    const ips = text.match(ipPattern) || [];
    
    // Basic validation
    return ips.filter(ip => {
      const parts = ip.split('.');
      return parts.every(part => {
        const num = parseInt(part);
        return num >= 0 && num <= 255;
      });
    });
  }

  /**
   * Scan text for any IOC matches
   */
  async scanText(text: string): Promise<{
    found: boolean;
    url_matches: IOCMatchResult[];
    domain_matches: IOCMatchResult[];
    ip_matches: IOCMatchResult[];
  }> {
    // Extract potential IOCs from text
    const urls = this.extractURLs(text);
    const domains = this.extractDomains(urls);
    const ips = this.extractIPs(text);

    // Match against database
    const urlMatches = await Promise.all(urls.map(url => this.matchURL(url)));
    const domainMatches = await Promise.all(domains.map(domain => this.matchDomain(domain)));
    const ipMatches = await Promise.all(ips.map(ip => this.matchIP(ip)));

    const found = 
      urlMatches.some(m => m.found) ||
      domainMatches.some(m => m.found) ||
      ipMatches.some(m => m.found);

    return {
      found,
      url_matches: urlMatches,
      domain_matches: domainMatches,
      ip_matches: ipMatches
    };
  }
}
