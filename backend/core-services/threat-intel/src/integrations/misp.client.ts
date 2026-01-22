import axios, { AxiosInstance } from 'axios';
import { logger } from '../utils/logger';

export interface MISPEvent {
  id: string;
  info: string;
  threat_level_id: string;
  analysis: string;
  date: string;
  published: boolean;
  Attribute: MISPAttribute[];
  Tag?: MISPTag[];
}

export interface MISPAttribute {
  id: string;
  event_id: string;
  category: string;
  type: string;
  value: string;
  comment?: string;
  to_ids: boolean;
  timestamp: string;
  Tag?: MISPTag[];
}

export interface MISPTag {
  name: string;
  colour?: string;
}

export interface MISPSearchParams {
  published?: boolean;
  publish_timestamp?: string;
  tags?: string[];
  type?: string;
  category?: string;
  include_context?: boolean;
  limit?: number;
}

export class MISPClient {
  private client: AxiosInstance;
  private baseUrl: string;
  private apiKey: string;
  private enabled: boolean;

  constructor() {
    this.baseUrl = process.env.MISP_URL || '';
    this.apiKey = process.env.MISP_API_KEY || '';
    this.enabled = Boolean(this.baseUrl && this.apiKey);

    if (!this.enabled) {
      logger.warn('MISP integration disabled - missing URL or API key');
      return;
    }

    this.client = axios.create({
      baseURL: this.baseUrl,
      headers: {
        'Authorization': this.apiKey,
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });

    logger.info(`MISP client initialized: ${this.baseUrl}`);
  }

  /**
   * Check if MISP integration is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Test MISP connection
   */
  async testConnection(): Promise<boolean> {
    if (!this.enabled) return false;

    try {
      const response = await this.client.get('/servers/getVersion');
      logger.info(`MISP connection successful, version: ${response.data.version}`);
      return true;
    } catch (error: any) {
      logger.error(`MISP connection failed: ${error.message}`);
      return false;
    }
  }

  /**
   * Fetch events from MISP
   */
  async fetchEvents(params: MISPSearchParams = {}): Promise<MISPEvent[]> {
    if (!this.enabled) {
      logger.debug('MISP is disabled, returning empty events');
      return [];
    }

    try {
      // Default to published events from last 7 days
      const searchParams = {
        published: params.published !== undefined ? params.published : true,
        publish_timestamp: params.publish_timestamp || '7d',
        include_context: params.include_context !== undefined ? params.include_context : true,
        limit: params.limit || 1000,
        ...params
      };

      logger.debug(`Fetching MISP events with params:`, searchParams);

      const response = await this.client.post('/events/restSearch', searchParams);

      if (response.data.response && Array.isArray(response.data.response)) {
        const events = response.data.response.map((item: any) => item.Event);
        logger.info(`Fetched ${events.length} events from MISP`);
        return events;
      }

      return [];
    } catch (error: any) {
      logger.error(`Failed to fetch MISP events: ${error.message}`);
      throw new Error(`MISP fetch failed: ${error.message}`);
    }
  }

  /**
   * Fetch specific event by ID
   */
  async fetchEvent(eventId: string): Promise<MISPEvent | null> {
    if (!this.enabled) return null;

    try {
      const response = await this.client.get(`/events/${eventId}`);
      return response.data.Event || null;
    } catch (error: any) {
      logger.error(`Failed to fetch MISP event ${eventId}: ${error.message}`);
      return null;
    }
  }

  /**
   * Search for attributes (IOCs)
   */
  async searchAttributes(params: {
    type?: string[];
    value?: string;
    category?: string[];
    to_ids?: boolean;
    limit?: number;
  }): Promise<MISPAttribute[]> {
    if (!this.enabled) return [];

    try {
      const searchParams = {
        returnFormat: 'json',
        ...params,
        limit: params.limit || 10000
      };

      const response = await this.client.post('/attributes/restSearch', searchParams);

      if (response.data.response && response.data.response.Attribute) {
        const attributes = response.data.response.Attribute;
        logger.info(`Found ${attributes.length} matching attributes in MISP`);
        return attributes;
      }

      return [];
    } catch (error: any) {
      logger.error(`Failed to search MISP attributes: ${error.message}`);
      throw new Error(`MISP attribute search failed: ${error.message}`);
    }
  }

  /**
   * Extract IOCs from MISP events
   */
  extractIOCs(events: MISPEvent[]): {
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

    for (const event of events) {
      if (!event.Attribute) continue;

      for (const attr of event.Attribute) {
        // Only include IOCs marked for detection
        if (!attr.to_ids) continue;

        switch (attr.type) {
          case 'url':
            iocs.urls.push(attr.value);
            break;
          case 'domain':
          case 'hostname':
            iocs.domains.push(attr.value);
            break;
          case 'ip-src':
          case 'ip-dst':
            iocs.ips.push(attr.value);
            break;
          case 'md5':
          case 'sha1':
          case 'sha256':
          case 'sha512':
            iocs.hashes.push(attr.value);
            break;
          case 'email':
          case 'email-src':
          case 'email-dst':
            iocs.emails.push(attr.value);
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

    logger.info(`Extracted IOCs from MISP: ${iocs.urls.length} URLs, ${iocs.domains.length} domains, ${iocs.ips.length} IPs`);

    return iocs;
  }

  /**
   * Check if a value matches any MISP IOC
   */
  async checkIOC(value: string, type: string): Promise<{
    found: boolean;
    attributes: MISPAttribute[];
  }> {
    if (!this.enabled) {
      return { found: false, attributes: [] };
    }

    try {
      const attributes = await this.searchAttributes({
        value,
        type: [type],
        to_ids: true,
        limit: 10
      });

      return {
        found: attributes.length > 0,
        attributes
      };
    } catch (error: any) {
      logger.error(`Failed to check IOC in MISP: ${error.message}`);
      return { found: false, attributes: [] };
    }
  }

  /**
   * Get threat level name
   */
  getThreatLevelName(levelId: string): string {
    const levels: Record<string, string> = {
      '1': 'High',
      '2': 'Medium',
      '3': 'Low',
      '4': 'Undefined'
    };
    return levels[levelId] || 'Unknown';
  }

  /**
   * Get analysis status name
   */
  getAnalysisName(analysisId: string): string {
    const statuses: Record<string, string> = {
      '0': 'Initial',
      '1': 'Ongoing',
      '2': 'Completed'
    };
    return statuses[analysisId] || 'Unknown';
  }
}
