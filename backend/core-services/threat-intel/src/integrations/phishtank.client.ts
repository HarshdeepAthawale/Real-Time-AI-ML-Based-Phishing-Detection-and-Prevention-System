import axios, { AxiosInstance } from 'axios';
import { logger } from '../utils/logger';
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import { promisify } from 'util';

const gunzip = promisify(zlib.gunzip);

export interface PhishTankEntry {
  phish_id: string;
  url: string;
  phish_detail_url: string;
  submission_time: string;
  verified: string;
  verification_time: string;
  online: string;
  target: string;
}

export class PhishTankClient {
  private client: AxiosInstance;
  private apiKey: string;
  private enabled: boolean;
  private baseUrl = 'https://checkurl.phishtank.com/checkurl/';
  private dataUrl = 'http://data.phishtank.com/data';
  private cacheDir: string;

  constructor() {
    this.apiKey = process.env.PHISHTANK_API_KEY || '';
    this.enabled = true; // PhishTank works without API key for some endpoints
    this.cacheDir = path.join(__dirname, '../../cache');

    // Create cache directory if it doesn't exist
    if (!fs.existsSync(this.cacheDir)) {
      fs.mkdirSync(this.cacheDir, { recursive: true });
    }

    this.client = axios.create({
      timeout: 30000,
      headers: {
        'User-Agent': 'phishing-detection-system/1.0'
      }
    });

    logger.info('PhishTank client initialized');
  }

  /**
   * Check if PhishTank integration is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Check if URL is in PhishTank database
   */
  async checkURL(url: string): Promise<{
    found: boolean;
    in_database: boolean;
    verified: boolean;
    phish_id?: string;
    detail_url?: string;
  }> {
    if (!this.enabled) {
      return { found: false, in_database: false, verified: false };
    }

    try {
      const formData = new URLSearchParams();
      formData.append('url', url);
      formData.append('format', 'json');
      if (this.apiKey) {
        formData.append('app_key', this.apiKey);
      }

      const response = await this.client.post(this.baseUrl, formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });

      const data = response.data.results;

      return {
        found: data.in_database,
        in_database: data.in_database,
        verified: data.verified,
        phish_id: data.phish_id,
        detail_url: data.phish_detail_page
      };
    } catch (error: any) {
      // Rate limiting is common with PhishTank
      if (error.response?.status === 509) {
        logger.warn('PhishTank rate limit exceeded');
        return { found: false, in_database: false, verified: false };
      }
      
      logger.error(`Failed to check URL in PhishTank: ${error.message}`);
      return { found: false, in_database: false, verified: false };
    }
  }

  /**
   * Download full PhishTank database
   */
  async downloadDatabase(): Promise<PhishTankEntry[]> {
    if (!this.enabled) return [];

    try {
      logger.info('Downloading PhishTank database...');

      // Use verified database (JSON format, compressed)
      const url = `${this.dataUrl}/online-valid.json.gz`;
      
      const response = await this.client.get(url, {
        responseType: 'arraybuffer',
        timeout: 120000 // 2 minutes for large file
      });

      // Decompress gzip
      const decompressed = await gunzip(Buffer.from(response.data));
      const jsonData = decompressed.toString('utf-8');
      const entries: PhishTankEntry[] = JSON.parse(jsonData);

      logger.info(`Downloaded ${entries.length} verified phishing URLs from PhishTank`);

      // Cache the data
      const cacheFile = path.join(this.cacheDir, 'phishtank_database.json');
      fs.writeFileSync(cacheFile, JSON.stringify({
        downloaded_at: new Date().toISOString(),
        count: entries.length,
        entries
      }));

      return entries;
    } catch (error: any) {
      logger.error(`Failed to download PhishTank database: ${error.message}`);
      
      // Try to load from cache if download fails
      return this.loadFromCache();
    }
  }

  /**
   * Load database from cache
   */
  private loadFromCache(): PhishTankEntry[] {
    try {
      const cacheFile = path.join(this.cacheDir, 'phishtank_database.json');
      
      if (fs.existsSync(cacheFile)) {
        const data = JSON.parse(fs.readFileSync(cacheFile, 'utf-8'));
        const ageHours = (Date.now() - new Date(data.downloaded_at).getTime()) / 3600000;
        
        if (ageHours < 24) {
          logger.info(`Loaded ${data.count} entries from PhishTank cache (${ageHours.toFixed(1)}h old)`);
          return data.entries;
        } else {
          logger.warn(`PhishTank cache is ${ageHours.toFixed(1)}h old, consider updating`);
          return data.entries;
        }
      }
    } catch (error: any) {
      logger.error(`Failed to load PhishTank cache: ${error.message}`);
    }
    
    return [];
  }

  /**
   * Get cached database or download if needed
   */
  async getDatabase(forceDownload: boolean = false): Promise<PhishTankEntry[]> {
    if (forceDownload) {
      return this.downloadDatabase();
    }

    // Try cache first
    const cached = this.loadFromCache();
    if (cached.length > 0) {
      return cached;
    }

    // Download if no cache
    return this.downloadDatabase();
  }

  /**
   * Extract URLs from database
   */
  extractURLs(entries: PhishTankEntry[]): string[] {
    const urls = entries
      .filter(entry => entry.verified === 'yes' && entry.online === 'yes')
      .map(entry => entry.url);
    
    return [...new Set(urls)];
  }

  /**
   * Extract domains from database
   */
  extractDomains(entries: PhishTankEntry[]): string[] {
    const domains = new Set<string>();

    for (const entry of entries) {
      if (entry.verified !== 'yes' || entry.online !== 'yes') continue;

      try {
        const url = new URL(entry.url);
        domains.add(url.hostname);
      } catch {
        // Invalid URL, skip
      }
    }

    return [...domains];
  }

  /**
   * Get statistics about the database
   */
  getDatabaseStats(entries: PhishTankEntry[]): {
    total: number;
    verified: number;
    online: number;
    targets: Record<string, number>;
  } {
    const stats = {
      total: entries.length,
      verified: 0,
      online: 0,
      targets: {} as Record<string, number>
    };

    for (const entry of entries) {
      if (entry.verified === 'yes') stats.verified++;
      if (entry.online === 'yes') stats.online++;
      
      if (entry.target) {
        stats.targets[entry.target] = (stats.targets[entry.target] || 0) + 1;
      }
    }

    return stats;
  }

  /**
   * Check if cache needs update
   */
  needsUpdate(): boolean {
    try {
      const cacheFile = path.join(this.cacheDir, 'phishtank_database.json');
      
      if (!fs.existsSync(cacheFile)) {
        return true;
      }

      const data = JSON.parse(fs.readFileSync(cacheFile, 'utf-8'));
      const ageHours = (Date.now() - new Date(data.downloaded_at).getTime()) / 3600000;
      
      // Update if older than 24 hours
      return ageHours > 24;
    } catch {
      return true;
    }
  }
}
