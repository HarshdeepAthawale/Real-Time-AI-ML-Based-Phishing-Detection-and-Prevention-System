import { DataSource, Repository } from 'typeorm';
import { SandboxAnalysis } from '../../../../shared/database/models/SandboxAnalysis';
import { Detection } from '../../../../shared/database/models/Detection';
import { Threat } from '../../../../shared/database/models/Threat';
import { logger } from '../utils/logger';
import axios, { AxiosInstance } from 'axios';

export class CorrelationService {
  private sandboxRepository: Repository<SandboxAnalysis>;
  private detectionRepository: Repository<Detection>;
  private threatRepository: Repository<Threat>;
  private detectionApiClient: AxiosInstance;
  
  constructor(
    dataSource: DataSource,
    detectionApiUrl?: string
  ) {
    this.sandboxRepository = dataSource.getRepository(SandboxAnalysis);
    this.detectionRepository = dataSource.getRepository(Detection);
    this.threatRepository = dataSource.getRepository(Threat);
    
    if (detectionApiUrl) {
      this.detectionApiClient = axios.create({
        baseURL: detectionApiUrl,
        timeout: 5000,
      });
    }
  }
  
  /**
   * Correlate sandbox analysis with detection API findings
   */
  async correlateWithDetections(analysisId: string): Promise<void> {
    try {
      const analysis = await this.sandboxRepository.findOne({
        where: { id: analysisId },
        relations: ['threat']
      });
      
      if (!analysis) {
        logger.warn(`Analysis ${analysisId} not found for correlation`);
        return;
      }
      
      // Find related detections based on URL or file hash
      let relatedDetections: Detection[] = [];
      
      if (analysis.target_url) {
        // Find detections with matching URL
        relatedDetections = await this.detectionRepository.find({
          where: {
            organization_id: analysis.organization_id,
            input_data: {
              url: analysis.target_url
            } as any
          },
          take: 10
        });
      } else if (analysis.target_file_hash) {
        // Find detections with matching file hash
        relatedDetections = await this.detectionRepository.find({
          where: {
            organization_id: analysis.organization_id,
            input_data: {
              fileHash: analysis.target_file_hash
            } as any
          },
          take: 10
        });
      }
      
      if (relatedDetections.length > 0) {
        logger.info(`Found ${relatedDetections.length} related detections for analysis ${analysisId}`);
        
        // Update threat confidence if sandbox confirms malicious behavior
        if (analysis.threat_id && analysis.result_data?.isMalicious) {
          await this.updateThreatConfidence(analysis.threat_id, analysis.result_data.threatScore);
        }
        
        // Link detections to threat if threat exists
        if (analysis.threat_id) {
          for (const detection of relatedDetections) {
            if (!detection.threat_id) {
              detection.threat_id = analysis.threat_id;
              await this.detectionRepository.save(detection);
            }
          }
        }
      }
      
    } catch (error: any) {
      logger.error(`Failed to correlate analysis ${analysisId} with detections`, error);
    }
  }
  
  /**
   * Update threat confidence score based on sandbox findings
   */
  private async updateThreatConfidence(threatId: string, sandboxScore: number): Promise<void> {
    try {
      const threat = await this.threatRepository.findOne({
        where: { id: threatId }
      });
      
      if (!threat) {
        return;
      }
      
      // Increase confidence if sandbox confirms malicious behavior
      // Weighted average: 70% original, 30% sandbox
      const newConfidence = Math.min(100, 
        (threat.confidence_score * 0.7) + (sandboxScore * 0.3)
      );
      
      threat.confidence_score = newConfidence;
      await this.threatRepository.save(threat);
      
      logger.info(`Updated threat ${threatId} confidence to ${newConfidence}`, {
        original: threat.confidence_score,
        sandboxScore
      });
    } catch (error: any) {
      logger.error(`Failed to update threat confidence for ${threatId}`, error);
    }
  }
  
  /**
   * Query detection API for related findings
   */
  async queryDetectionAPI(query: { url?: string; fileHash?: string }): Promise<any> {
    if (!this.detectionApiClient) {
      return null;
    }
    
    try {
      const response = await this.detectionApiClient.post('/api/v1/detect', {
        url: query.url,
        fileHash: query.fileHash
      });
      
      return response.data;
    } catch (error: any) {
      logger.warn('Failed to query detection API', error.message);
      return null;
    }
  }
}
