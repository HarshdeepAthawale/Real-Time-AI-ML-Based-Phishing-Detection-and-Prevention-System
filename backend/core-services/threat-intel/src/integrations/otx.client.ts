import axios, { AxiosInstance } from 'axios';
import { logger } from '../utils/logger';

export interface OTXPulse {
  id: string;
  name: string;
  description: string;
  author_name: string;
  created: string;
  modified: string;
  tags: string[];
  references: string[];
  indicators: OTXIndicator[];
  tlp?: string;
  adversary?: string;
  targeted_countries?: string[];
}

export interface OTXIndicator {
  id: string;
  indicator: string;
  type: string;
  title?: string;
  description?: string;
  created: string;
  is_active: number;
}

export interface OTXSearchResult {
  pulses: OTXPulse[];
  count: number;
}

export class OTXClient {
  private client: AxiosInstance;
  private apiKey: string;
  private enabled: boolean;
  private baseUrl = 'https://otx.alienvault.com';

  constructor(apiKey?: string) {
    this.apiKey = apiKey || process.env.OTX_API_KEY || '';
    this.enabled = Boolean(this.apiKey);

    if (!this.enabled) {
      logger.warn('OTX integration disabled - missing API key');
      return;
    }

    this.client = axios.create({
      baseURL: this.baseUrl,
      headers: {
        'X-OTX-API-KEY': this.apiKey,
        'Accept': 'application/json'
      },
      timeout: 30000
    });

    logger.info('OTX client initialized');
  }

  /**
   * Check if OTX integration is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Test OTX connection
   */
  async testConnection(): Promise<boolean> {
    if (!this.enabled) return false;

    try {
      const response = await this.client.get('/api/v1/user/me');
      logger.info(`OTX connection successful, user: ${response.data.username}`);
      return true;
    } catch (error: any) {
      logger.error(`OTX connection failed: ${error.message}`);
      return false;
    }
  }

  /**
   * Get subscribed pulses (threat feeds)
   */
  async getSubscribedPulses(params: {
    modified_since?: string;
    limit?: number;
    page?: number;
  } = {}): Promise<OTXSearchResult> {
    if (!this.enabled) {
      return { pulses: [], count: 0 };
    }

    try {
      const queryParams = new URLSearchParams();
      if (params.modified_since) queryParams.append('modified_since', params.modified_since);
      if (params.limit) queryParams.append('limit', params.limit.toString());
      if (params.page) queryParams.append('page', params.page.toString());

      const response = await this.client.get(`/api/v1/pulses/subscribed?${queryParams}`);
      
      const pulses = response.data.results || [];
      const count = response.data.count || 0;

      logger.info(`Fetched ${pulses.length} subscribed pulses from OTX (total: ${count})`);
      
      return { pulses, count };
    } catch (error: any) {
      logger.error(`Failed to fetch OTX pulses: ${error.message}`);
      throw new Error(`OTX fetch failed: ${error.message}`);
    }
  }

  /**
   * Get pulses by tag
   */
  async getPulsesByTag(tag: string, limit: number = 50): Promise<OTXPulse[]> {
    if (!this.enabled) return [];

    try {
      const response = await this.client.get(`/api/v1/pulses/activity?limit=${limit}&tag=${encodeURIComponent(tag)}`);
      const pulses = response.data.results || [];
      
      logger.info(`Fetched ${pulses.length} pulses for tag "${tag}" from OTX`);
      return pulses;
    } catch (error: any) {
      logger.error(`Failed to fetch OTX pulses by tag: ${error.message}`);
      return [];
    }
  }

  /**
   * Get specific pulse by ID
   */
  async getPulse(pulseId: string): Promise<OTXPulse | null> {
    if (!this.enabled) return null;

    try {
      const response = await this.client.get(`/api/v1/pulses/${pulseId}`);
      return response.data || null;
    } catch (error: any) {
      logger.error(`Failed to fetch OTX pulse ${pulseId}: ${error.message}`);
      return null;
    }
  }

  /**
   * Check URL reputation
   */
  async checkURL(url: string): Promise<{
    found: boolean;
    pulses: OTXPulse[];
    reputation: number;
  }> {
    if (!this.enabled) {
      return { found: false, pulses: [], reputation: 0 };
    }

    try {
      const encodedUrl = encodeURIComponent(url);
      const response = await this.client.get(`/api/v1/indicators/url/${encodedUrl}/general`);
      
      const pulses = response.data.pulse_info?.pulses || [];
      const reputation = response.data.reputation || 0;

      return {
        found: pulses.length > 0,
        pulses,
        reputation
      };
    } catch (error: any) {
      logger.error(`Failed to check URL in OTX: ${error.message}`);
      return { found: false, pulses: [], reputation: 0 };
    }
  }

  /**
   * Check domain reputation
   */
  async checkDomain(domain: string): Promise<{
    found: boolean;
    pulses: OTXPulse[];
    reputation: number;
  }> {
    if (!this.enabled) {
      return { found: false, pulses: [], reputation: 0 };
    }

    try {
      const response = await this.client.get(`/api/v1/indicators/domain/${domain}/general`);
      
      const pulses = response.data.pulse_info?.pulses || [];
      const reputation = response.data.reputation || 0;

      return {
        found: pulses.length > 0,
        pulses,
        reputation
      };
    } catch (error: any) {
      logger.error(`Failed to check domain in OTX: ${error.message}`);
      return { found: false, pulses: [], reputation: 0 };
    }
  }

  /**
   * Check IP reputation
   */
  async checkIP(ip: string): Promise<{
    found: boolean;
    pulses: OTXPulse[];
    reputation: number;
  }> {
    if (!this.enabled) {
      return { found: false, pulses: [], reputation: 0 };
    }

    try {
      const response = await this.client.get(`/api/v1/indicators/IPv4/${ip}/general`);
      
      const pulses = response.data.pulse_info?.pulses || [];
      const reputation = response.data.reputation || 0;

      return {
        found: pulses.length > 0,
        pulses,
        reputation
      };
    } catch (error: any) {
      logger.error(`Failed to check IP in OTX: ${error.message}`);
      return { found: false, pulses: [], reputation: 0 };
    }
  }

  /**
   * Extract IOCs from pulses
   */
  extractIOCs(pulses: OTXPulse[]): {
    urls: string[];
    domains: string[];
    ips: string[];
    hashes: string[];
    emails: string[];
  } {
    const iocs = {
      urls: [] as string[],
      domains: [] as string[],
      ips: [] as string[],
      hashes: [] as string[],
      emails: [] as string[]
    };

    for (const pulse of pulses) {
      if (!pulse.indicators) continue;

      for (const indicator of pulse.indicators) {
        // Only include active indicators
        if (indicator.is_active !== 1) continue;

        const value = indicator.indicator;

        switch (indicator.type) {
          case 'URL':
            iocs.urls.push(value);
            break;
          case 'domain':
          case 'hostname':
            iocs.domains.push(value);
            break;
          case 'IPv4':
          case 'IPv6':
            iocs.ips.push(value);
            break;
          case 'FileHash-MD5':
          case 'FileHash-SHA1':
          case 'FileHash-SHA256':
            iocs.hashes.push(value);
            break;
          case 'email':
            iocs.emails.push(value);
            break;
        }
      }
    }

    // Deduplicate
    iocs.urls = [...new Set(iocs.urls)];
    iocs.domains = [...new Set(iocs.domains)];
    iocs.ips = [...new Set(iocs.ips)];
    iocs.hashes = [...new Set(iocs.hashes)];
    iocs.emails = [...new Set(iocs.emails)];

    logger.info(`Extracted IOCs from OTX: ${iocs.urls.length} URLs, ${iocs.domains.length} domains, ${iocs.ips.length} IPs`);

    return iocs;
  }

  /**
   * Get recent pulses (last N days)
   */
  async getRecentPulses(days: number = 7, limit: number = 100): Promise<OTXPulse[]> {
    if (!this.enabled) return [];

    try {
      const modifiedSince = new Date();
      modifiedSince.setDate(modifiedSince.getDate() - days);
      const modifiedSinceStr = modifiedSince.toISOString();

      const result = await this.getSubscribedPulses({
        modified_since: modifiedSinceStr,
        limit
      });

      return result.pulses;
    } catch (error: any) {
      logger.error(`Failed to fetch recent OTX pulses: ${error.message}`);
      return [];
    }
  }

  /**
   * Get malware-related pulses
   */
  async getMalwarePulses(limit: number = 50): Promise<OTXPulse[]> {
    const tags = ['malware', 'phishing', 'ransomware', 'trojan'];
    const allPulses: OTXPulse[] = [];

    for (const tag of tags) {
      const pulses = await this.getPulsesByTag(tag, limit);
      allPulses.push(...pulses);
    }

    // Deduplicate by ID
    const uniquePulses = Array.from(
      new Map(allPulses.map(p => [p.id, p])).values()
    );

    return uniquePulses;
  }
}
