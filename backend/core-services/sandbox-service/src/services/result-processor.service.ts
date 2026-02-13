import { BaseSandboxClient, SandboxResult } from '../integrations/base-sandbox.client';
import { DataSource, Repository } from 'typeorm';
import { SandboxAnalysis } from '../../../../shared/database/models/SandboxAnalysis';
import { Threat } from '../../../../shared/database/models/Threat';
import { BehavioralAnalyzerService, BehavioralIndicators } from './behavioral-analyzer.service';
import { CorrelationService } from './correlation.service';
import { logger } from '../utils/logger';

export class ResultProcessorService {
  private sandboxClient: BaseSandboxClient;
  private sandboxRepository: Repository<SandboxAnalysis>;
  private threatRepository: Repository<Threat>;
  private behavioralAnalyzer: BehavioralAnalyzerService;
  private correlationService?: CorrelationService;
  
  constructor(
    sandboxClient: BaseSandboxClient,
    dataSource: DataSource,
    behavioralAnalyzer: BehavioralAnalyzerService,
    correlationService?: CorrelationService
  ) {
    this.sandboxClient = sandboxClient;
    this.sandboxRepository = dataSource.getRepository(SandboxAnalysis);
    this.threatRepository = dataSource.getRepository(Threat);
    this.behavioralAnalyzer = behavioralAnalyzer;
    this.correlationService = correlationService;
  }
  
  async processResults(analysisId: string): Promise<void> {
    try {
      // Get analysis record
      const analysis = await this.sandboxRepository.findOne({
        where: { id: analysisId }
      });
      
      if (!analysis) {
        throw new Error(`Analysis ${analysisId} not found`);
      }
      
      if (!analysis.sandbox_job_id) {
        logger.warn(`Analysis ${analysisId} has no sandbox job ID`);
        return;
      }
      
      // Get sandbox results
      const sandboxResult = await this.sandboxClient.getResults(analysis.sandbox_job_id);
      
      // Update status
      analysis.status = sandboxResult.status;
      
      if (sandboxResult.status === 'running' && !analysis.started_at) {
        analysis.started_at = new Date();
      }
      
      if (sandboxResult.status === 'completed') {
        analysis.completed_at = new Date();
        
        // Analyze behavioral indicators
        const behavioralAnalysis = this.behavioralAnalyzer.analyze(sandboxResult);
        
        // Update analysis record with results
        analysis.result_data = {
          ...analysis.result_data,
          sandbox: sandboxResult.results,
          behavioral: behavioralAnalysis,
          isMalicious: behavioralAnalysis.isMalicious,
          threatScore: behavioralAnalysis.threatScore,
          completed_at: new Date().toISOString()
        };
        
        // If malicious, create threat record
        if (behavioralAnalysis.isMalicious) {
          await this.createThreatRecord(analysis, behavioralAnalysis, sandboxResult);
        }
      } else if (sandboxResult.status === 'failed') {
        analysis.result_data = {
          ...analysis.result_data,
          error: sandboxResult.error || 'Sandbox analysis failed',
          failed_at: new Date().toISOString()
        };
      }
      
      await this.sandboxRepository.save(analysis);
      
      // Correlate with detection API if analysis is completed and malicious
      if (sandboxResult.status === 'completed' && analysis.result_data?.isMalicious && this.correlationService) {
        try {
          await this.correlationService.correlateWithDetections(analysisId);
        } catch (error: any) {
          logger.warn(`Failed to correlate analysis ${analysisId}`, error);
          // Don't fail the entire processing if correlation fails
        }
      }
      
      logger.info(`Processed results for analysis ${analysisId}`, {
        status: sandboxResult.status,
        isMalicious: sandboxResult.status === 'completed' ? 
          (sandboxResult.results ? this.behavioralAnalyzer.analyze(sandboxResult).isMalicious : false) : undefined
      });
      
    } catch (error: any) {
      logger.error(`Failed to process results for analysis ${analysisId}`, error);
      
      // Update analysis record with error
      try {
        const analysis = await this.sandboxRepository.findOne({
          where: { id: analysisId }
        });
        
        if (analysis) {
          analysis.status = 'failed';
          analysis.result_data = {
            ...analysis.result_data,
            error: error.message || 'Unknown error',
            failed_at: new Date().toISOString()
          };
          await this.sandboxRepository.save(analysis);
        }
      } catch (updateError) {
        logger.error('Failed to update analysis record with error', updateError);
      }
      
      throw error;
    }
  }
  
  private async createThreatRecord(
    analysis: SandboxAnalysis,
    behavioralAnalysis: BehavioralIndicators,
    sandboxResult: SandboxResult
  ): Promise<void> {
    // Check if threat already exists for this analysis
    if (analysis.threat_id) {
      logger.info(`Threat already exists for analysis ${analysis.id}`);
      return;
    }
    
    // Determine threat type based on indicators
    let threatType = 'malware';
    if (behavioralAnalysis.indicators.some(i => i.toLowerCase().includes('c2') || i.toLowerCase().includes('communication'))) {
      threatType = 'c2_malware';
    } else if (behavioralAnalysis.indicators.some(i => i.toLowerCase().includes('exfiltrat') || i.toLowerCase().includes('data'))) {
      threatType = 'data_stealer';
    } else if (behavioralAnalysis.indicators.some(i => i.toLowerCase().includes('injection'))) {
      threatType = 'trojan';
    }
    
    // Determine severity based on threat score
    let severity = 'low';
    if (behavioralAnalysis.threatScore >= 80) {
      severity = 'critical';
    } else if (behavioralAnalysis.threatScore >= 60) {
      severity = 'high';
    } else if (behavioralAnalysis.threatScore >= 40) {
      severity = 'medium';
    }
    
    // Create threat record
    const threat = this.threatRepository.create({
      organization_id: analysis.organization_id,
      threat_type: threatType,
      severity,
      status: 'detected',
      confidence_score: behavioralAnalysis.threatScore,
      source: 'sandbox',
      source_value: analysis.target_url || analysis.target_file_hash || 'unknown',
      title: `Sandbox Analysis: ${threatType}`,
      description: `Malicious behavior detected in sandbox analysis. Indicators: ${behavioralAnalysis.indicators.join(', ')}`,
      metadata: {
        analysis_id: analysis.id,
        sandbox_provider: analysis.sandbox_provider,
        sandbox_job_id: analysis.sandbox_job_id,
        indicators: behavioralAnalysis.indicators,
        networkActivity: behavioralAnalysis.categories?.network,
        fileSystemActivity: behavioralAnalysis.categories?.filesystem,
        processActivity: behavioralAnalysis.categories?.process
      },
      detected_at: new Date()
    });
    
    const savedThreat = await this.threatRepository.save(threat);
    
    // Link threat to analysis
    analysis.threat_id = savedThreat.id;
    await this.sandboxRepository.save(analysis);
    
    logger.info(`Created threat record for analysis ${analysis.id}`, {
      threatId: savedThreat.id,
      threatType,
      severity
    });
  }
}
