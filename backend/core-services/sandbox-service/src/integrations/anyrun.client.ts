import axios, { AxiosInstance } from 'axios';
import FormData from 'form-data';
import { BaseSandboxClient, SandboxResult, NetworkActivity, FileSystemActivity, ProcessActivity, SandboxSubmitOptions } from './base-sandbox.client';
import { logger } from '../utils/logger';

export class AnyRunClient extends BaseSandboxClient {
  private client: AxiosInstance;
  private apiKey: string;
  
  constructor(apiKey: string) {
    super();
    this.apiKey = apiKey;
    this.client = axios.create({
      baseURL: 'https://api.any.run/v1',
      headers: {
        'Authorization': `API-Key ${apiKey}`,
      },
      timeout: 30000,
    });
  }
  
  async submitFile(fileBuffer: Buffer, filename: string, options?: SandboxSubmitOptions): Promise<string> {
    try {
      const formData = new FormData();
      formData.append('file', fileBuffer, filename);
      
      if (options?.timeout) {
        formData.append('timeout', options.timeout.toString());
      }
      
      const response = await this.client.post('/analysis', formData, {
        headers: {
          ...formData.getHeaders(),
        },
      });
      
      return response.data.data.taskid;
    } catch (error: any) {
      logger.error('Any.run file submission failed', { error: error.message, filename });
      throw new Error(`Any.run file submission failed: ${error.message}`);
    }
  }
  
  async submitURL(url: string, options?: SandboxSubmitOptions): Promise<string> {
    try {
      const params: any = {
        obj_type: 'url',
        obj_url: url,
      };
      
      if (options?.timeout) {
        params.timeout = options.timeout;
      }
      
      const response = await this.client.post('/analysis', params);
      return response.data.data.taskid;
    } catch (error: any) {
      logger.error('Any.run URL submission failed', { error: error.message, url });
      throw new Error(`Any.run URL submission failed: ${error.message}`);
    }
  }
  
  async getStatus(jobId: string): Promise<SandboxResult> {
    try {
      const response = await this.client.get(`/analysis/${jobId}`);
      const data = response.data.data;
      
      let status: 'pending' | 'running' | 'completed' | 'failed' = 'pending';
      if (data.status === 'done') {
        status = 'completed';
      } else if (data.status === 'running' || data.status === 'processing') {
        status = 'running';
      } else if (data.status === 'failed' || data.status === 'error') {
        status = 'failed';
      }
      
      return {
        jobId,
        status,
        analysisId: jobId,
      };
    } catch (error: any) {
      logger.error('Any.run status check failed', { error: error.message, jobId });
      throw new Error(`Any.run status check failed: ${error.message}`);
    }
  }
  
  async getResults(analysisId: string): Promise<SandboxResult> {
    try {
      const response = await this.client.get(`/analysis/${analysisId}`);
      const data = response.data.data;
      
      if (data.status !== 'done') {
        return {
          jobId: analysisId,
          status: data.status === 'running' ? 'running' : 'pending',
          analysisId,
        };
      }
      
      return {
        jobId: analysisId,
        status: 'completed',
        analysisId,
        results: {
          network: this.extractNetworkActivity(data.network),
          filesystem: this.extractFileSystemActivity(data.files),
          processes: this.extractProcessActivity(data.processes),
          screenshots: data.screenshots?.map((s: any) => s.url || s),
          signatures: data.threats?.map((t: any) => t.name || t) || [],
          score: this.calculateScore(data.threats, data.verdict)
        }
      };
    } catch (error: any) {
      logger.error('Any.run results fetch failed', { error: error.message, analysisId });
      throw new Error(`Any.run results fetch failed: ${error.message}`);
    }
  }
  
  private extractNetworkActivity(network: any): NetworkActivity[] {
    if (!network) return [];
    
    const activities: NetworkActivity[] = [];
    
    if (network.connections) {
      network.connections.forEach((conn: any) => {
        activities.push({
          protocol: conn.protocol || 'tcp',
          destination: conn.ip || conn.host || '',
          port: conn.port || 0,
        });
      });
    }
    
    if (network.http) {
      network.http.forEach((http: any) => {
        activities.push({
          protocol: 'http',
          destination: http.host || http.ip || '',
          port: http.port || 80,
          method: http.method,
          path: http.path || http.uri,
          statusCode: http.status
        });
      });
    }
    
    return activities;
  }
  
  private extractFileSystemActivity(files: any): FileSystemActivity[] {
    if (!files || !Array.isArray(files)) return [];
    
    return files.map((file: any) => ({
      action: file.action || 'created',
      path: file.path || file.name || '',
      fileType: file.type || file.mime
    }));
  }
  
  private extractProcessActivity(processes: any[]): ProcessActivity[] {
    if (!processes || !Array.isArray(processes)) return [];
    
    return processes.map((proc: any) => ({
      name: proc.name || '',
      pid: proc.pid || 0,
      commandLine: proc.cmdline || proc.command || '',
      parentPid: proc.ppid || proc.parent_pid
    }));
  }
  
  private calculateScore(threats: any[], verdict: string): number {
    if (!threats || threats.length === 0) {
      return verdict === 'malicious' ? 8 : 0;
    }
    
    // Any.run threat score is typically 0-10
    // Map to 0-100 scale
    const maxThreatScore = 10;
    const threatCount = threats.length;
    const baseScore = threatCount * 10;
    
    return Math.min(100, baseScore);
  }
}
