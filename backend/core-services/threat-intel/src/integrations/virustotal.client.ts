import axios, { AxiosInstance } from 'axios';
import { logger } from '../utils/logger';

export interface VirusTotalAnalysis {
  id: string;
  type: string;
  attributes: {
    status: string;
    stats: {
      harmless: number;
      malicious: number;
      suspicious: number;
      undetected: number;
      timeout: number;
    };
    results: Record<string, {
      category: string;
      result: string;
      method: string;
      engine_name: string;
    }>;
    last_analysis_date: number;
  };
}

export interface VirusTotalURLReport {
  data: VirusTotalAnalysis;
}

export interface VirusTotalDomainReport {
  data: {
    id: string;
    type: string;
    attributes: {
      last_analysis_stats: {
        harmless: number;
        malicious: number;
        suspicious: number;
        undetected: number;
        timeout: number;
      };
      reputation: number;
      categories: Record<string, string>;
      last_analysis_date: number;
    };
  };
}

export class VirusTotalClient {
  private client: AxiosInstance;
  private apiKey: string;
  private enabled: boolean;
  private baseUrl = 'https://www.virustotal.com/api/v3';
  private requestCount = 0;
  private lastRequestTime = Date.now();
  private readonly maxRequestsPerMinute = 4; // Free tier limit

  constructor() {
    this.apiKey = process.env.VIRUSTOTAL_API_KEY || '';
    this.enabled = Boolean(this.apiKey);

    if (!this.enabled) {
      logger.warn('VirusTotal integration disabled - missing API key');
      return;
    }

    this.client = axios.create({
      baseURL: this.baseUrl,
      headers: {
        'x-apikey': this.apiKey,
        'Accept': 'application/json'
      },
      timeout: 30000
    });

    logger.info('VirusTotal client initialized');
  }

  /**
   * Check if VirusTotal integration is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Rate limiting for free tier (4 requests/minute)
   */
  private async rateLimit(): Promise<void> {
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;

    // Reset counter if more than 1 minute has passed
    if (timeSinceLastRequest > 60000) {
      this.requestCount = 0;
      this.lastRequestTime = now;
    }

    // Wait if we've hit the limit
    if (this.requestCount >= this.maxRequestsPerMinute) {
      const waitTime = 60000 - timeSinceLastRequest;
      if (waitTime > 0) {
        logger.debug(`VirusTotal rate limit reached, waiting ${waitTime}ms`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
        this.requestCount = 0;
        this.lastRequestTime = Date.now();
      }
    }

    this.requestCount++;
  }

  /**
   * Submit URL for scanning
   */
  async scanURL(url: string): Promise<{ id: string }> {
    if (!this.enabled) {
      throw new Error('VirusTotal is not enabled');
    }

    await this.rateLimit();

    try {
      const formData = new URLSearchParams();
      formData.append('url', url);

      const response = await this.client.post('/urls', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });

      const analysisId = response.data.data.id;
      logger.info(`Submitted URL to VirusTotal: ${analysisId}`);

      return { id: analysisId };
    } catch (error: any) {
      logger.error(`Failed to submit URL to VirusTotal: ${error.message}`);
      throw new Error(`VirusTotal scan failed: ${error.message}`);
    }
  }

  /**
   * Get URL analysis report
   */
  async getURLReport(url: string): Promise<VirusTotalURLReport | null> {
    if (!this.enabled) return null;

    await this.rateLimit();

    try {
      // Encode URL as base64url (without padding)
      const urlId = Buffer.from(url).toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=/g, '');

      const response = await this.client.get<VirusTotalURLReport>(`/urls/${urlId}`);
      
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        logger.debug(`URL not found in VirusTotal: ${url}`);
        return null;
      }
      logger.error(`Failed to get URL report from VirusTotal: ${error.message}`);
      return null;
    }
  }

  /**
   * Get domain report
   */
  async getDomainReport(domain: string): Promise<VirusTotalDomainReport | null> {
    if (!this.enabled) return null;

    await this.rateLimit();

    try {
      const response = await this.client.get<VirusTotalDomainReport>(`/domains/${domain}`);
      
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        logger.debug(`Domain not found in VirusTotal: ${domain}`);
        return null;
      }
      logger.error(`Failed to get domain report from VirusTotal: ${error.message}`);
      return null;
    }
  }

  /**
   * Get IP address report
   */
  async getIPReport(ip: string): Promise<any> {
    if (!this.enabled) return null;

    await this.rateLimit();

    try {
      const response = await this.client.get(`/ip_addresses/${ip}`);
      
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) {
        logger.debug(`IP not found in VirusTotal: ${ip}`);
        return null;
      }
      logger.error(`Failed to get IP report from VirusTotal: ${error.message}`);
      return null;
    }
  }

  /**
   * Get analysis status
   */
  async getAnalysisStatus(analysisId: string): Promise<{
    status: string;
    completed: boolean;
  }> {
    if (!this.enabled) {
      return { status: 'unknown', completed: false };
    }

    await this.rateLimit();

    try {
      const response = await this.client.get(`/analyses/${analysisId}`);
      const status = response.data.data.attributes.status;

      return {
        status,
        completed: status === 'completed'
      };
    } catch (error: any) {
      logger.error(`Failed to get analysis status from VirusTotal: ${error.message}`);
      return { status: 'error', completed: false };
    }
  }

  /**
   * Wait for analysis to complete
   */
  async waitForAnalysis(analysisId: string, maxWaitSeconds: number = 60): Promise<VirusTotalAnalysis | null> {
    const startTime = Date.now();
    const maxWaitTime = maxWaitSeconds * 1000;

    while (Date.now() - startTime < maxWaitTime) {
      const status = await this.getAnalysisStatus(analysisId);

      if (status.completed) {
        try {
          const response = await this.client.get<{ data: VirusTotalAnalysis }>(`/analyses/${analysisId}`);
          return response.data.data;
        } catch (error: any) {
          logger.error(`Failed to get completed analysis: ${error.message}`);
          return null;
        }
      }

      // Wait 10 seconds before checking again
      await new Promise(resolve => setTimeout(resolve, 10000));
    }

    logger.warn(`Analysis ${analysisId} did not complete within ${maxWaitSeconds}s`);
    return null;
  }

  /**
   * Check if URL is malicious
   */
  async checkURL(url: string): Promise<{
    found: boolean;
    malicious: boolean;
    stats: {
      harmless: number;
      malicious: number;
      suspicious: number;
      undetected: number;
    };
    positives: number;
    total: number;
  }> {
    if (!this.enabled) {
      return {
        found: false,
        malicious: false,
        stats: { harmless: 0, malicious: 0, suspicious: 0, undetected: 0 },
        positives: 0,
        total: 0
      };
    }

    try {
      const report = await this.getURLReport(url);

      if (!report) {
        // URL not in database, submit for scanning
        await this.scanURL(url);
        return {
          found: false,
          malicious: false,
          stats: { harmless: 0, malicious: 0, suspicious: 0, undetected: 0 },
          positives: 0,
          total: 0
        };
      }

      const stats = report.data.attributes.stats;
      const total = stats.harmless + stats.malicious + stats.suspicious + stats.undetected;
      const positives = stats.malicious + stats.suspicious;

      return {
        found: true,
        malicious: stats.malicious > 0,
        stats,
        positives,
        total
      };
    } catch (error: any) {
      logger.error(`Failed to check URL in VirusTotal: ${error.message}`);
      return {
        found: false,
        malicious: false,
        stats: { harmless: 0, malicious: 0, suspicious: 0, undetected: 0 },
        positives: 0,
        total: 0
      };
    }
  }

  /**
   * Check if domain is malicious
   */
  async checkDomain(domain: string): Promise<{
    found: boolean;
    malicious: boolean;
    reputation: number;
    stats: {
      harmless: number;
      malicious: number;
      suspicious: number;
      undetected: number;
    };
  }> {
    if (!this.enabled) {
      return {
        found: false,
        malicious: false,
        reputation: 0,
        stats: { harmless: 0, malicious: 0, suspicious: 0, undetected: 0 }
      };
    }

    try {
      const report = await this.getDomainReport(domain);

      if (!report) {
        return {
          found: false,
          malicious: false,
          reputation: 0,
          stats: { harmless: 0, malicious: 0, suspicious: 0, undetected: 0 }
        };
      }

      const stats = report.data.attributes.last_analysis_stats;
      const reputation = report.data.attributes.reputation || 0;

      return {
        found: true,
        malicious: stats.malicious > 0,
        reputation,
        stats
      };
    } catch (error: any) {
      logger.error(`Failed to check domain in VirusTotal: ${error.message}`);
      return {
        found: false,
        malicious: false,
        reputation: 0,
        stats: { harmless: 0, malicious: 0, suspicious: 0, undetected: 0 }
      };
    }
  }

  /**
   * Get detection percentage
   */
  getDetectionPercentage(stats: { harmless: number; malicious: number; suspicious: number; undetected: number }): number {
    const total = stats.harmless + stats.malicious + stats.suspicious + stats.undetected;
    if (total === 0) return 0;

    const positives = stats.malicious + stats.suspicious;
    return (positives / total) * 100;
  }
}
