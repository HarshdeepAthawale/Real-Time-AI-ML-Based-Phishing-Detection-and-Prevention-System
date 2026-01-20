import axios, { AxiosInstance } from 'axios';
import { config } from '../config';
import { logger } from '../utils/logger';
import { DetectionRequest, MLServiceResponse } from '../types';

export class OrchestratorService {
  private nlpClient: AxiosInstance;
  private urlClient: AxiosInstance;
  private visualClient: AxiosInstance;
  
  constructor() {
    this.nlpClient = axios.create({
      baseURL: config.mlServices.nlp,
      timeout: 5000
    });
    
    this.urlClient = axios.create({
      baseURL: config.mlServices.url,
      timeout: 10000
    });
    
    this.visualClient = axios.create({
      baseURL: config.mlServices.visual,
      timeout: 30000
    });
  }
  
  async analyzeEmail(request: DetectionRequest): Promise<MLServiceResponse> {
    const startTime = Date.now();
    
    try {
      // Parallel calls to NLP and URL services
      const [nlpResult, urlResults] = await Promise.all([
        this.nlpClient.post('/api/v1/analyze-email', {
          raw_email: request.emailContent,
          include_features: request.includeFeatures || false
        }).catch(err => {
          logger.warn('NLP service error in email analysis', err.message);
          return { data: null };
        }),
        this.extractAndAnalyzeURLs(request.emailContent || '')
      ]);
      
      const processingTime = Date.now() - startTime;
      
      return {
        nlp: nlpResult.data,
        url: urlResults.length > 0 ? urlResults : null,
        visual: null, // Visual analysis is async for emails
        processingTimeMs: processingTime
      };
    } catch (error: any) {
      logger.error('Error in email analysis orchestration', error);
      throw error;
    }
  }
  
  async analyzeURL(request: DetectionRequest): Promise<MLServiceResponse> {
    const startTime = Date.now();
    
    try {
      // Parallel calls to URL and Visual services
      const [urlResult, visualResult] = await Promise.all([
        this.urlClient.post('/api/v1/analyze-url', {
          url: request.url,
          legitimate_domain: request.legitimateDomain
        }).catch(err => {
          logger.warn('URL service error', err.message);
          return { data: null };
        }),
        this.visualClient.post('/api/v1/analyze-page', {
          url: request.url,
          legitimate_url: request.legitimateUrl
        }).catch(err => {
          // Visual analysis can fail, don't block URL analysis
          logger.warn('Visual analysis failed', err.message);
          return { data: null };
        })
      ]);
      
      // Extract text from URL page for NLP analysis
      const pageText = visualResult.data?.dom_analysis?.text || '';
      let nlpResult = null;
      
      if (pageText) {
        try {
          nlpResult = await this.nlpClient.post('/api/v1/analyze-text', {
            text: pageText,
            include_features: false
          });
        } catch (err: any) {
          logger.warn('NLP analysis failed', err.message);
        }
      }
      
      const processingTime = Date.now() - startTime;
      
      return {
        nlp: nlpResult?.data || null,
        url: urlResult.data,
        visual: visualResult.data,
        processingTimeMs: processingTime
      };
    } catch (error: any) {
      logger.error('Error in URL analysis orchestration', error);
      throw error;
    }
  }
  
  async analyzeText(request: DetectionRequest): Promise<MLServiceResponse> {
    const startTime = Date.now();
    
    try {
      const nlpResult = await this.nlpClient.post('/api/v1/analyze-text', {
        text: request.text,
        include_features: request.includeFeatures || false
      }).catch(err => {
        logger.warn('NLP service error', err.message);
        return { data: null };
      });
      
      const processingTime = Date.now() - startTime;
      
      return {
        nlp: nlpResult.data,
        url: null,
        visual: null,
        processingTimeMs: processingTime
      };
    } catch (error: any) {
      logger.error('Error in text analysis orchestration', error);
      throw error;
    }
  }
  
  private async extractAndAnalyzeURLs(emailContent: string): Promise<any[]> {
    // Extract URLs from email
    const urlPattern = /https?:\/\/[^\s<>"{}|\\^`\[\]]+/g;
    const urls = emailContent.match(urlPattern) || [];
    
    // Analyze each URL (limit to first 5 for performance)
    const urlAnalyses = await Promise.allSettled(
      urls.slice(0, 5).map(url => 
        this.urlClient.post('/api/v1/analyze-url', { url })
      )
    );
    
    return urlAnalyses
      .filter(result => result.status === 'fulfilled')
      .map(result => (result as PromiseFulfilledResult<any>).value.data);
  }
}
