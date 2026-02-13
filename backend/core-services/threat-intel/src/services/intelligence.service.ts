import { MISPClient } from '../integrations/misp.client';
import { OTXClient } from '../integrations/otx.client';
import { PhishTankClient } from '../integrations/phishtank.client';
import { URLhausClient } from '../integrations/urlhaus.client';
import { VirusTotalClient } from '../integrations/virustotal.client';
import { IOCMatcherService } from './ioc-matcher.service';
import { getPostgreSQL } from '../../../../shared/database/connection';
import { IOC as IOCEntity } from '../../../../shared/database/models/IOC';
import { ThreatIntelligenceFeed } from '../../../../shared/database/models/ThreatIntelligenceFeed';
import { logger } from '../utils/logger';

export interface ThreatCheckResult {
  is_threat: boolean;
  confidence: number;
  sources: string[];
  details: {
    misp?: any;
    otx?: any;
    phishtank?: any;
    urlhaus?: any;
    virustotal?: any;
    ioc_matches?: any;
  };
  threat_level: 'high' | 'medium' | 'low' | 'none';
}

export class IntelligenceService {
  private mispClient: MISPClient;
  private otxClient: OTXClient;
  private phishtankClient: PhishTankClient;
  private urlhausClient: URLhausClient;
  private virusTotalClient: VirusTotalClient;
  private iocMatcher: IOCMatcherService;

  constructor() {
    this.mispClient = new MISPClient();
    this.otxClient = new OTXClient();
    this.phishtankClient = new PhishTankClient();
    this.urlhausClient = new URLhausClient();
    this.virusTotalClient = new VirusTotalClient();
    this.iocMatcher = new IOCMatcherService();

    logger.info('Intelligence Service initialized');
  }

  /**
   * Initialize the service (build Bloom filters, test connections)
   */
  async initialize(): Promise<void> {
    logger.info('Initializing Intelligence Service...');

    try {
      // Test connections to all enabled feeds
      const connections = await Promise.all([
        this.mispClient.isEnabled() ? this.mispClient.testConnection() : Promise.resolve(false),
        this.otxClient.isEnabled() ? this.otxClient.testConnection() : Promise.resolve(false)
      ]);

      logger.info('Feed connections tested:', {
        misp: connections[0],
        otx: connections[1],
        phishtank: this.phishtankClient.isEnabled(),
        urlhaus: this.urlhausClient.isEnabled(),
        virustotal: this.virusTotalClient.isEnabled()
      });

      // Build Bloom filters from database
      await this.iocMatcher.buildBloomFilters();

      logger.info('Intelligence Service initialized successfully');
    } catch (error: any) {
      logger.error(`Failed to initialize Intelligence Service: ${error.message}`);
      throw error;
    }
  }

  /**
   * Check URL against all threat intelligence sources
   */
  async checkURL(url: string): Promise<ThreatCheckResult> {
    const result: ThreatCheckResult = {
      is_threat: false,
      confidence: 0,
      sources: [],
      details: {},
      threat_level: 'none'
    };

    try {
      // Fast local IOC check first
      const iocMatch = await this.iocMatcher.matchURL(url);
      if (iocMatch.found) {
        result.is_threat = true;
        result.sources.push('local_ioc');
        result.details.ioc_matches = iocMatch.matches;
        
        // Calculate confidence based on IOC sources
        const avgConfidence = iocMatch.matches.reduce((sum, m) => sum + m.confidence, 0) / iocMatch.matches.length;
        result.confidence = Math.max(result.confidence, avgConfidence);
      }

      // Check external sources in parallel
      const [phishtankResult, urlhausResult, otxResult] = await Promise.allSettled([
        this.phishtankClient.checkURL(url),
        this.urlhausClient.checkURL(url),
        this.otxClient.checkURL(url)
      ]);

      // PhishTank
      if (phishtankResult.status === 'fulfilled' && phishtankResult.value.found) {
        result.is_threat = true;
        result.sources.push('phishtank');
        result.details.phishtank = phishtankResult.value;
        result.confidence = Math.max(result.confidence, 0.9);
      }

      // URLhaus
      if (urlhausResult.status === 'fulfilled' && urlhausResult.value.found) {
        result.is_threat = true;
        result.sources.push('urlhaus');
        result.details.urlhaus = urlhausResult.value;
        result.confidence = Math.max(result.confidence, 0.95);
      }

      // OTX
      if (otxResult.status === 'fulfilled' && otxResult.value.found) {
        result.is_threat = true;
        result.sources.push('otx');
        result.details.otx = otxResult.value;
        result.confidence = Math.max(result.confidence, 0.85);
      }

      // VirusTotal (rate limited, use sparingly)
      if (!result.is_threat && this.virusTotalClient.isEnabled()) {
        try {
          const vtResult = await this.virusTotalClient.checkURL(url);
          if (vtResult.found && vtResult.malicious) {
            result.is_threat = true;
            result.sources.push('virustotal');
            result.details.virustotal = vtResult;
            result.confidence = Math.max(result.confidence, vtResult.positives / vtResult.total);
          }
        } catch (error: any) {
          logger.debug(`VirusTotal check skipped: ${error.message}`);
        }
      }

      // Determine threat level
      result.threat_level = this.calculateThreatLevel(result.confidence, result.sources.length);

      logger.info(`URL check complete: ${url} - Threat: ${result.is_threat}, Confidence: ${result.confidence.toFixed(2)}, Sources: ${result.sources.join(', ')}`);

    } catch (error: any) {
      logger.error(`Failed to check URL: ${error.message}`);
    }

    return result;
  }

  /**
   * Check domain against threat intelligence
   */
  async checkDomain(domain: string): Promise<ThreatCheckResult> {
    const result: ThreatCheckResult = {
      is_threat: false,
      confidence: 0,
      sources: [],
      details: {},
      threat_level: 'none'
    };

    try {
      // Local IOC check
      const iocMatch = await this.iocMatcher.matchDomain(domain);
      if (iocMatch.found) {
        result.is_threat = true;
        result.sources.push('local_ioc');
        result.details.ioc_matches = iocMatch.matches;
        const avgConfidence = iocMatch.matches.reduce((sum, m) => sum + m.confidence, 0) / iocMatch.matches.length;
        result.confidence = Math.max(result.confidence, avgConfidence);
      }

      // Check external sources
      const [otxResult, urlhausResult] = await Promise.allSettled([
        this.otxClient.checkDomain(domain),
        this.urlhausClient.checkHost(domain)
      ]);

      if (otxResult.status === 'fulfilled' && otxResult.value.found) {
        result.is_threat = true;
        result.sources.push('otx');
        result.details.otx = otxResult.value;
        result.confidence = Math.max(result.confidence, 0.8);
      }

      if (urlhausResult.status === 'fulfilled' && urlhausResult.value.found) {
        result.is_threat = true;
        result.sources.push('urlhaus');
        result.details.urlhaus = urlhausResult.value;
        result.confidence = Math.max(result.confidence, 0.9);
      }

      result.threat_level = this.calculateThreatLevel(result.confidence, result.sources.length);

    } catch (error: any) {
      logger.error(`Failed to check domain: ${error.message}`);
    }

    return result;
  }

  /**
   * Check IP against threat intelligence
   */
  async checkIP(ip: string): Promise<ThreatCheckResult> {
    const result: ThreatCheckResult = {
      is_threat: false,
      confidence: 0,
      sources: [],
      details: {},
      threat_level: 'none'
    };

    try {
      // Local IOC check
      const iocMatch = await this.iocMatcher.matchIP(ip);
      if (iocMatch.found) {
        result.is_threat = true;
        result.sources.push('local_ioc');
        result.details.ioc_matches = iocMatch.matches;
        const avgConfidence = iocMatch.matches.reduce((sum, m) => sum + m.confidence, 0) / iocMatch.matches.length;
        result.confidence = Math.max(result.confidence, avgConfidence);
      }

      // OTX check
      const otxResult = await this.otxClient.checkIP(ip);
      if (otxResult.found) {
        result.is_threat = true;
        result.sources.push('otx');
        result.details.otx = otxResult;
        result.confidence = Math.max(result.confidence, 0.8);
      }

      result.threat_level = this.calculateThreatLevel(result.confidence, result.sources.length);

    } catch (error: any) {
      logger.error(`Failed to check IP: ${error.message}`);
    }

    return result;
  }

  /**
   * Sync IOCs from all sources
   */
  async syncAllFeeds(): Promise<{
    success: boolean;
    feeds_synced: string[];
    total_iocs: number;
    errors: string[];
  }> {
    logger.info('Starting feed synchronization...');
    
    const feedsSynced: string[] = [];
    const errors: string[] = [];
    let totalIOCs = 0;

    // Sync MISP
    if (this.mispClient.isEnabled()) {
      try {
        await this.syncMISP();
        feedsSynced.push('MISP');
      } catch (error: any) {
        errors.push(`MISP: ${error.message}`);
        logger.error(`MISP sync failed: ${error.message}`);
      }
    }

    // Sync OTX
    if (this.otxClient.isEnabled()) {
      try {
        await this.syncOTX();
        feedsSynced.push('OTX');
      } catch (error: any) {
        errors.push(`OTX: ${error.message}`);
        logger.error(`OTX sync failed: ${error.message}`);
      }
    }

    // Sync PhishTank
    try {
      const count = await this.syncPhishTank();
      totalIOCs += count;
      feedsSynced.push('PhishTank');
    } catch (error: any) {
      errors.push(`PhishTank: ${error.message}`);
      logger.error(`PhishTank sync failed: ${error.message}`);
    }

    // Sync URLhaus
    try {
      const count = await this.syncURLhaus();
      totalIOCs += count;
      feedsSynced.push('URLhaus');
    } catch (error: any) {
      errors.push(`URLhaus: ${error.message}`);
      logger.error(`URLhaus sync failed: ${error.message}`);
    }

    // Rebuild Bloom filters after sync
    try {
      await this.iocMatcher.buildBloomFilters();
      logger.info('Bloom filters rebuilt after sync');
    } catch (error: any) {
      logger.error(`Failed to rebuild Bloom filters: ${error.message}`);
    }

    const success = feedsSynced.length > 0;
    
    logger.info(`Feed sync complete: ${feedsSynced.length}/${feedsSynced.length + errors.length} successful, ${totalIOCs} IOCs synced`);

    return {
      success,
      feeds_synced: feedsSynced,
      total_iocs: totalIOCs,
      errors
    };
  }

  /**
   * Sync MISP events
   */
  private async syncMISP(): Promise<number> {
    const events = await this.mispClient.fetchEvents({ publish_timestamp: '7d', limit: 100 });
    const iocs = this.mispClient.extractIOCs(events);
    
    return await this.saveIOCs('MISP', iocs);
  }

  /**
   * Sync OTX pulses
   */
  private async syncOTX(): Promise<number> {
    const pulses = await this.otxClient.getRecentPulses(7, 100);
    const iocs = this.otxClient.extractIOCs(pulses);
    
    return await this.saveIOCs('OTX', iocs);
  }

  /**
   * Sync PhishTank database
   */
  private async syncPhishTank(): Promise<number> {
    const entries = await this.phishtankClient.getDatabase();
    const urls = this.phishtankClient.extractURLs(entries);
    const domains = this.phishtankClient.extractDomains(entries);
    
    return await this.saveIOCs('PhishTank', { urls, domains, ips: [], hashes: [], emails: [] });
  }

  /**
   * Sync URLhaus database
   */
  private async syncURLhaus(): Promise<number> {
    const entries = await this.urlhausClient.getRecentURLs(1000);
    const extracted = this.urlhausClient.extractIOCs(entries);
    const iocs = {
      urls: extracted.urls || [],
      domains: extracted.domains || [],
      ips: [] as string[],
      hashes: [] as string[],
      emails: [] as string[],
    };
    return await this.saveIOCs('URLhaus', iocs);
  }

  /**
   * Save IOCs to database
   */
  private async saveIOCs(source: string, iocs: {
    urls: string[];
    domains: string[];
    ips: string[];
    hashes: string[];
    emails: string[];
  }): Promise<number> {
    try {
      const dataSource = getPostgreSQL();
      const iocRepository = dataSource.getRepository(IOCEntity);
      
      let count = 0;
      const now = new Date();

      // Save URLs
      for (const url of iocs.urls) {
        await iocRepository.upsert({
          ioc_type: 'url',
          ioc_value: url,
          source,
          is_active: true,
          first_seen_at: now,
          last_seen_at: now,
          confidence: 0.8
        }, ['ioc_type', 'ioc_value', 'source']);
        count++;
      }

      // Save domains
      for (const domain of iocs.domains) {
        await iocRepository.upsert({
          ioc_type: 'domain',
          ioc_value: domain,
          source,
          is_active: true,
          first_seen_at: now,
          last_seen_at: now,
          confidence: 0.75
        }, ['ioc_type', 'ioc_value', 'source']);
        count++;
      }

      // Save IPs
      for (const ip of iocs.ips) {
        await iocRepository.upsert({
          ioc_type: 'ip',
          ioc_value: ip,
          source,
          is_active: true,
          first_seen_at: now,
          last_seen_at: now,
          confidence: 0.7
        }, ['ioc_type', 'ioc_value', 'source']);
        count++;
      }

      // Save hashes
      for (const hash of iocs.hashes) {
        await iocRepository.upsert({
          ioc_type: 'hash',
          ioc_value: hash,
          source,
          is_active: true,
          first_seen_at: now,
          last_seen_at: now,
          confidence: 0.9
        }, ['ioc_type', 'ioc_value', 'source']);
        count++;
      }

      // Save emails
      for (const email of iocs.emails) {
        await iocRepository.upsert({
          ioc_type: 'email',
          ioc_value: email,
          source,
          is_active: true,
          first_seen_at: now,
          last_seen_at: now,
          confidence: 0.65
        }, ['ioc_type', 'ioc_value', 'source']);
        count++;
      }

      // Update feed sync status
      const feedRepository = dataSource.getRepository(ThreatIntelligenceFeed);
      await feedRepository.update(
        { name: source },
        {
          last_sync_at: now,
          last_sync_status: 'success',
          iocs_imported: count
        }
      );

      logger.info(`Saved ${count} IOCs from ${source}`);
      return count;

    } catch (error: any) {
      logger.error(`Failed to save IOCs from ${source}: ${error.message}`);
      throw error;
    }
  }

  /**
   * Calculate threat level based on confidence and sources
   */
  private calculateThreatLevel(confidence: number, sourceCount: number): 'high' | 'medium' | 'low' | 'none' {
    if (confidence >= 0.8 && sourceCount >= 2) return 'high';
    if (confidence >= 0.6 || sourceCount >= 2) return 'medium';
    if (confidence >= 0.4 || sourceCount >= 1) return 'low';
    return 'none';
  }

  /**
   * Get service statistics
   */
  getStats() {
    return {
      feeds: {
        misp: this.mispClient.isEnabled(),
        otx: this.otxClient.isEnabled(),
        phishtank: this.phishtankClient.isEnabled(),
        urlhaus: this.urlhausClient.isEnabled(),
        virustotal: this.virusTotalClient.isEnabled()
      },
      bloom_filters: this.iocMatcher.getStats()
    };
  }
}
