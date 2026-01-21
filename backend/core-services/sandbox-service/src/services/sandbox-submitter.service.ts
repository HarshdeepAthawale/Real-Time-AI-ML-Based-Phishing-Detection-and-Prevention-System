import { BaseSandboxClient } from '../integrations/base-sandbox.client';
import { FileAnalyzerService, FileAnalysis } from './file-analyzer.service';
import { DataSource, Repository } from 'typeorm';
import { SandboxAnalysis } from '../../../../shared/database/models/SandboxAnalysis';
import { logger } from '../utils/logger';
import { config } from '../config';

export class SandboxSubmitterService {
  private sandboxClient: BaseSandboxClient;
  private fileAnalyzer: FileAnalyzerService;
  private sandboxRepository: Repository<SandboxAnalysis>;
  
  constructor(
    sandboxClient: BaseSandboxClient,
    fileAnalyzer: FileAnalyzerService,
    dataSource: DataSource
  ) {
    this.sandboxClient = sandboxClient;
    this.fileAnalyzer = fileAnalyzer;
    this.sandboxRepository = dataSource.getRepository(SandboxAnalysis);
  }
  
  async submitFile(
    fileBuffer: Buffer,
    filename: string,
    organizationId: string
  ): Promise<string> {
    // Analyze file first
    const analysis = await this.fileAnalyzer.analyzeFile(fileBuffer, filename);
    
    // Check if file requires sandbox analysis
    if (!analysis.requiresSandbox) {
      logger.info(`File ${filename} does not require sandbox analysis`);
      // Still create record but mark as not requiring sandbox
      const record = await this.createAnalysisRecord(analysis, null, organizationId, 'file');
      return record.id;
    }
    
    // Submit to sandbox
    const jobId = await this.sandboxClient.submitFile(fileBuffer, filename, {
      timeout: config.timeout
    });
    
    // Create analysis record
    const record = await this.createAnalysisRecord(analysis, jobId, organizationId, 'file');
    
    logger.info(`File submitted to sandbox: ${jobId}`, { analysisId: record.id });
    return record.id;
  }
  
  async submitURL(url: string, organizationId: string): Promise<string> {
    // Submit URL to sandbox
    const jobId = await this.sandboxClient.submitURL(url, {
      timeout: config.timeout
    });
    
    // Create analysis record
    const record = this.sandboxRepository.create({
      organization_id: organizationId,
      analysis_type: 'url',
      target_url: url,
      sandbox_provider: config.provider,
      sandbox_job_id: jobId,
      status: 'pending',
      submitted_at: new Date(),
      result_data: {
        url,
        submitted_at: new Date().toISOString()
      }
    });
    
    const saved = await this.sandboxRepository.save(record);
    
    logger.info(`URL submitted to sandbox: ${jobId}`, { analysisId: saved.id });
    return saved.id;
  }
  
  private async createAnalysisRecord(
    fileAnalysis: FileAnalysis,
    jobId: string | null,
    organizationId: string,
    analysisType: 'file'
  ): Promise<SandboxAnalysis> {
    const record = this.sandboxRepository.create({
      organization_id: organizationId,
      analysis_type: analysisType,
      target_file_hash: fileAnalysis.hash.sha256,
      sandbox_provider: jobId ? config.provider : null,
      sandbox_job_id: jobId,
      status: jobId ? 'pending' : 'not_required',
      submitted_at: new Date(),
      result_data: {
        filename: fileAnalysis.filename,
        fileType: fileAnalysis.fileType,
        mimeType: fileAnalysis.mimeType,
        size: fileAnalysis.size,
        hash: fileAnalysis.hash,
        metadata: fileAnalysis.metadata,
        extractedText: fileAnalysis.extractedText?.substring(0, 1000), // Limit text length
        isExecutable: fileAnalysis.isExecutable,
        requiresSandbox: fileAnalysis.requiresSandbox,
        submitted_at: new Date().toISOString()
      }
    });
    
    return await this.sandboxRepository.save(record);
  }
}
