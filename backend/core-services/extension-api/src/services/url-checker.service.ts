import axios, { AxiosInstance } from 'axios';
import { config } from '../config';
import { CacheService } from './cache.service';
import { PrivacyFilterService } from './privacy-filter.service';
import { logger } from '../utils/logger';

export interface URLCheckOptions {
  includeFullAnalysis?: boolean;
  privacyMode?: boolean;
  pageText?: string;
  pageTitle?: string;
  screenshot?: string; // base64 encoded
}

export interface URLCheckResult {
  isThreat: boolean;
  severity?: string;
  confidence?: number;
  cached: boolean;
  warningMessage?: string;
  details?: {
    url_analysis?: any;
    text_analysis?: any;
    visual_analysis?: any;
  };
  timestamp: string;
}

export class URLCheckerService {
  private detectionApiClient: AxiosInstance;
  private cache: CacheService;
  private privacyFilter: PrivacyFilterService;
  
  constructor(
    detectionApiUrl: string,
    cache: CacheService,
    privacyFilter: PrivacyFilterService
  ) {
    this.cache = cache;
    this.privacyFilter = privacyFilter;
    
    this.detectionApiClient = axios.create({
      baseURL: detectionApiUrl,
      timeout: config.detectionApi.timeout.url,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }
  
  async checkURL(url: string, options: URLCheckOptions = {}): Promise<URLCheckResult> {
    // Check cache first
    const cacheKey = this.cache.generateURLKey(url);
    const cached = await this.cache.get(cacheKey);
    
    if (cached) {
      logger.debug('Cache hit for URL check', { url: this.sanitizeURL(url) });
      return {
        ...cached,
        cached: true
      };
    }
    
    // Privacy filter - only send minimal data if privacy mode
    let urlToCheck = url;
    if (options.privacyMode) {
      urlToCheck = this.privacyFilter.filterURL(url);
    }
    
    try {
      // Prepare detection API request
      const detectionRequest: any = {
        url: urlToCheck
      };
      
      // Add optional fields
      if (options.pageText) {
        detectionRequest.text = options.privacyMode 
          ? this.privacyFilter.extractTextContent(options.pageText)
          : options.pageText;
      }
      
      if (options.screenshot) {
        detectionRequest.image = options.screenshot;
      }
      
      if (options.includeFullAnalysis) {
        detectionRequest.includeFeatures = true;
      }
      
      // Call detection API
      logger.debug('Calling detection API for URL check', { 
        url: this.sanitizeURL(urlToCheck),
        hasText: !!options.pageText,
        hasScreenshot: !!options.screenshot
      });
      
      const response = await this.detectionApiClient.post('/api/v1/detect/url', detectionRequest, {
        headers: {
          'X-API-Key': process.env.DETECTION_API_KEY || '', // Optional API key
        }
      });
      
      const detectionResult = response.data;
      
      const result: URLCheckResult = {
        isThreat: detectionResult.isThreat || false,
        severity: detectionResult.severity,
        confidence: detectionResult.confidence || detectionResult.overall_confidence || 0,
        cached: false,
        warningMessage: detectionResult.isThreat 
          ? 'This page may be a phishing attempt. Proceed with caution.'
          : undefined,
        details: {
          url_analysis: detectionResult.sources?.url,
          text_analysis: detectionResult.sources?.nlp,
          visual_analysis: detectionResult.sources?.visual
        },
        timestamp: new Date().toISOString()
      };
      
      // Cache result (shorter TTL for extension API)
      await this.cache.set(cacheKey, {
        isThreat: result.isThreat,
        severity: result.severity,
        confidence: result.confidence,
        warningMessage: result.warningMessage,
        details: result.details
      }, config.cache.urlTtl);
      
      logger.info('URL check completed', {
        url: this.sanitizeURL(url),
        isThreat: result.isThreat,
        confidence: result.confidence
      });
      
      return result;
    } catch (error: any) {
      logger.error('URL check failed', {
        url: this.sanitizeURL(url),
        error: error.message,
        status: error.response?.status
      });
      
      // Return safe default on error
      return {
        isThreat: false,
        cached: false,
        timestamp: new Date().toISOString()
      };
    }
  }
  
  /**
   * Sanitize URL for logging (remove sensitive parts)
   */
  private sanitizeURL(url: string): string {
    try {
      const parsed = new URL(url);
      return `${parsed.protocol}//${parsed.hostname}${parsed.pathname.substring(0, 50)}`;
    } catch {
      return url.substring(0, 100);
    }
  }
}
