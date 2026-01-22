import axios, { AxiosInstance } from 'axios';
import { logger } from '../utils/logger';

export interface URLhausEntry {
  id: string;
  urlhaus_reference: string;
  url: string;
  url_status: string;
  host: string;
  date_added: string;
  threat: string;
  blacklists: {
    spamhaus_dbl: string;
    surbl: string;
  };
  reporter: string;
  larted: boolean;
  tags: string[];
}

export interface URLhausResponse {
  query_status: string;
  data?: URLhausEntry[];
  urls?: URLhausEntry[];
}

export class URLhausClient {
  private client: AxiosInstance;
  private enabled: boolean;
  private baseUrl = 'https://urlhaus-api.abuse.ch/v1';

  constructor() {
    this.enabled = true; // URLhaus is public API, no key needed

    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'phishing-detection-system/1.0'
      }
    });

    logger.info('URLhaus client initialized');
  }

  /**
   * Check if URLhaus integration is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Check if URL exists in URLhaus
   */
  async checkURL(url: string): Promise<{
    found: boolean;
    status?: string;
    threat?: string;
    entry?: URLhausEntry;
  }> {
    if (!this.enabled) {
      return { found: false };
    }

    try {
      const formData = new URLSearchParams();
      formData.append('url', url);

      const response = await this.client.post<URLhausResponse>('/url/', formData);

      if (response.data.query_status === 'ok' && response.data.data) {
        const entry = response.data.data[0];
        return {
          found: true,
          status: entry.url_status,
          threat: entry.threat,
          entry
        };
      }

      return { found: false };
    } catch (error: any) {
      logger.error(`Failed to check URL in URLhaus: ${error.message}`);
      return { found: false };
    }
  }

  /**
   * Check if host/domain exists in URLhaus
   */
  async checkHost(host: string): Promise<{
    found: boolean;
    urls: URLhausEntry[];
  }> {
    if (!this.enabled) {
      return { found: false, urls: [] };
    }

    try {
      const formData = new URLSearchParams();
      formData.append('host', host);

      const response = await this.client.post<URLhausResponse>('/host/', formData);

      if (response.data.query_status === 'ok' && response.data.urls) {
        return {
          found: true,
          urls: response.data.urls
        };
      }

      return { found: false, urls: [] };
    } catch (error: any) {
      logger.error(`Failed to check host in URLhaus: ${error.message}`);
      return { found: false, urls: [] };
    }
  }

  /**
   * Get recent URLs (last N days)
   */
  async getRecentURLs(limit: number = 1000): Promise<URLhausEntry[]> {
    if (!this.enabled) return [];

    try {
      const formData = new URLSearchParams();
      formData.append('limit', limit.toString());

      const response = await this.client.post<URLhausResponse>('/urls/recent/', formData);

      if (response.data.query_status === 'ok' && response.data.urls) {
        logger.info(`Fetched ${response.data.urls.length} recent URLs from URLhaus`);
        return response.data.urls;
      }

      return [];
    } catch (error: any) {
      logger.error(`Failed to fetch recent URLs from URLhaus: ${error.message}`);
      return [];
    }
  }

  /**
   * Get URLs by tag
   */
  async getURLsByTag(tag: string): Promise<URLhausEntry[]> {
    if (!this.enabled) return [];

    try {
      const formData = new URLSearchParams();
      formData.append('tag', tag);

      const response = await this.client.post<URLhausResponse>('/tag/', formData);

      if (response.data.query_status === 'ok' && response.data.urls) {
        logger.info(`Fetched ${response.data.urls.length} URLs for tag "${tag}" from URLhaus`);
        return response.data.urls;
      }

      return [];
    } catch (error: any) {
      logger.error(`Failed to fetch URLs by tag from URLhaus: ${error.message}`);
      return [];
    }
  }

  /**
   * Download full URLhaus database (CSV format)
   */
  async downloadDatabase(): Promise<URLhausEntry[]> {
    if (!this.enabled) return [];

    try {
      logger.info('Downloading URLhaus database...');

      // Download CSV format (faster than JSON)
      const response = await this.client.get('https://urlhaus.abuse.ch/downloads/csv_recent/', {
        timeout: 120000, // 2 minutes
        responseType: 'text'
      });

      const entries = this.parseCSV(response.data);
      logger.info(`Downloaded ${entries.length} URLs from URLhaus`);

      return entries;
    } catch (error: any) {
      logger.error(`Failed to download URLhaus database: ${error.message}`);
      return [];
    }
  }

  /**
   * Parse URLhaus CSV format
   */
  private parseCSV(csv: string): URLhausEntry[] {
    const entries: URLhausEntry[] = [];
    const lines = csv.split('\n');

    for (const line of lines) {
      // Skip comments and empty lines
      if (line.startsWith('#') || line.trim() === '') continue;

      const parts = line.split(',').map(p => p.replace(/"/g, ''));
      
      if (parts.length < 8) continue;

      try {
        entries.push({
          id: parts[0],
          urlhaus_reference: parts[1],
          url: parts[2],
          url_status: parts[3],
          host: parts[4] || '',
          date_added: parts[1],
          threat: parts[5] || '',
          blacklists: {
            spamhaus_dbl: parts[6] || 'not listed',
            surbl: parts[7] || 'not listed'
          },
          reporter: parts[8] || '',
          larted: false,
          tags: parts[9] ? parts[9].split('|') : []
        });
      } catch (error) {
        // Skip malformed lines
        continue;
      }
    }

    return entries;
  }

  /**
   * Extract IOCs from URLhaus entries
   */
  extractIOCs(entries: URLhausEntry[]): {
    urls: string[];
    domains: string[];
  } {
    const urls = new Set<string>();
    const domains = new Set<string>();

    for (const entry of entries) {
      // Only include online/active URLs
      if (entry.url_status !== 'online') continue;

      urls.add(entry.url);
      
      if (entry.host) {
        domains.add(entry.host);
      }
    }

    logger.info(`Extracted IOCs from URLhaus: ${urls.size} URLs, ${domains.size} domains`);

    return {
      urls: [...urls],
      domains: [...domains]
    };
  }

  /**
   * Get URLs by threat type
   */
  async getMalwareURLs(threatTypes: string[] = ['malware_download']): Promise<URLhausEntry[]> {
    const allUrls: URLhausEntry[] = [];

    for (const threatType of threatTypes) {
      const urls = await this.getURLsByTag(threatType);
      allUrls.push(...urls);
    }

    // Deduplicate by ID
    const uniqueUrls = Array.from(
      new Map(allUrls.map(u => [u.id, u])).values()
    );

    return uniqueUrls;
  }

  /**
   * Get statistics about entries
   */
  getStats(entries: URLhausEntry[]): {
    total: number;
    online: number;
    offline: number;
    threats: Record<string, number>;
    tags: Record<string, number>;
  } {
    const stats = {
      total: entries.length,
      online: 0,
      offline: 0,
      threats: {} as Record<string, number>,
      tags: {} as Record<string, number>
    };

    for (const entry of entries) {
      if (entry.url_status === 'online') stats.online++;
      if (entry.url_status === 'offline') stats.offline++;

      if (entry.threat) {
        stats.threats[entry.threat] = (stats.threats[entry.threat] || 0) + 1;
      }

      if (entry.tags) {
        for (const tag of entry.tags) {
          stats.tags[tag] = (stats.tags[tag] || 0) + 1;
        }
      }
    }

    return stats;
  }
}
