import axios, { AxiosInstance } from 'axios';
import { config } from '../config';
import { PrivacyFilterService } from './privacy-filter.service';
import { URLCheckerService } from './url-checker.service';
import { logger } from '../utils/logger';

export interface EmailScanOptions {
  privacyMode?: boolean;
  scanLinks?: boolean;
  includeFullAnalysis?: boolean;
}

export interface EmailScanResult {
  isThreat: boolean;
  threatType?: string;
  severity?: string;
  confidence?: number;
  suspiciousLinks?: string[];
  details?: {
    email_analysis?: any;
    link_analysis?: any;
  };
  timestamp: string;
}

export class EmailScannerService {
  private detectionApiClient: AxiosInstance;
  private privacyFilter: PrivacyFilterService;
  private urlChecker: URLCheckerService | null;
  
  constructor(
    detectionApiUrl: string,
    privacyFilter: PrivacyFilterService,
    urlChecker?: URLCheckerService
  ) {
    this.privacyFilter = privacyFilter;
    this.urlChecker = urlChecker || null;
    
    this.detectionApiClient = axios.create({
      baseURL: detectionApiUrl,
      timeout: config.detectionApi.timeout.email,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
  
  async scanEmail(
    emailContent: string,
    options: EmailScanOptions = {}
  ): Promise<EmailScanResult> {
    try {
      // Filter email content if privacy mode
      let contentToScan: string;
      if (options.privacyMode) {
        const filtered = this.privacyFilter.filterEmailContent(emailContent);
        // Use subject and from for privacy mode scanning
        contentToScan = `Subject: ${filtered.subject}\nFrom: ${filtered.from}`;
      } else {
        // Extract text content from email
        contentToScan = this.privacyFilter.extractTextContent(emailContent);
      }
      
      logger.debug('Scanning email', {
        privacyMode: options.privacyMode,
        contentLength: contentToScan.length,
        scanLinks: options.scanLinks
      });
      
      // Call detection API for email analysis
      const response = await this.detectionApiClient.post('/api/v1/detect/email', {
        emailContent: contentToScan,
        includeFullAnalysis: options.includeFullAnalysis || false
      }, {
        headers: {
          'X-API-Key': process.env.DETECTION_API_KEY || '', // Optional API key
        }
      });
      
      const detectionResult = response.data;
      
      // Extract suspicious links if requested
      let suspiciousLinks: string[] = [];
      if (options.scanLinks && this.urlChecker) {
        suspiciousLinks = await this.scanEmailLinks(emailContent, options.privacyMode || false);
      }
      
      const result: EmailScanResult = {
        isThreat: detectionResult.isThreat || false,
        threatType: detectionResult.threatType,
        severity: detectionResult.severity,
        confidence: detectionResult.confidence || detectionResult.overall_confidence || 0,
        suspiciousLinks: suspiciousLinks.length > 0 ? suspiciousLinks : undefined,
        details: {
          email_analysis: detectionResult.sources?.nlp,
          link_analysis: suspiciousLinks.length > 0 ? { count: suspiciousLinks.length } : undefined
        },
        timestamp: new Date().toISOString()
      };
      
      logger.info('Email scan completed', {
        isThreat: result.isThreat,
        confidence: result.confidence,
        suspiciousLinksCount: suspiciousLinks.length
      });
      
      return result;
    } catch (error: any) {
      logger.error('Email scan failed', {
        error: error.message,
        status: error.response?.status
      });
      
      return {
        isThreat: false,
        timestamp: new Date().toISOString()
      };
    }
  }
  
  /**
   * Scan links found in email content
   */
  private async scanEmailLinks(
    emailContent: string,
    privacyMode: boolean
  ): Promise<string[]> {
    const suspiciousLinks: string[] = [];
    
    try {
      // Extract links from email
      const linkPattern = /https?:\/\/[^\s<>"{}|\\^`\[\]]+/g;
      const links = emailContent.match(linkPattern) || [];
      
      logger.debug('Scanning email links', { linkCount: links.length });
      
      // Check each link (limit to 10 links to avoid timeout)
      const linksToCheck = links.slice(0, 10);
      
      for (const link of linksToCheck) {
        try {
          if (this.urlChecker) {
            const linkCheck = await this.urlChecker.checkURL(link, {
              privacyMode,
              includeFullAnalysis: false
            });
            
            if (linkCheck.isThreat) {
              suspiciousLinks.push(link);
            }
          }
        } catch (error: any) {
          logger.warn('Link check failed', { link, error: error.message });
          // Continue checking other links
        }
      }
    } catch (error: any) {
      logger.error('Failed to scan email links', { error: error.message });
    }
    
    return suspiciousLinks;
  }
}
