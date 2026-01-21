import axios, { AxiosInstance } from 'axios';
import FormData from 'form-data';
import { BaseSandboxClient, SandboxResult, NetworkActivity, FileSystemActivity, ProcessActivity, SandboxSubmitOptions } from './base-sandbox.client';
import { logger } from '../utils/logger';

export class CuckooClient extends BaseSandboxClient {
  private client: AxiosInstance;
  private baseURL: string;
  
  constructor(baseURL: string, apiKey?: string) {
    super();
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL: `${baseURL}/api`,
      headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {},
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
      
      const response = await this.client.post('/tasks/create/file', formData, {
        headers: {
          ...formData.getHeaders(),
        },
      });
      
      return response.data.task_id.toString();
    } catch (error: any) {
      logger.error('Cuckoo file submission failed', { error: error.message, filename });
      throw new Error(`Cuckoo file submission failed: ${error.message}`);
    }
  }
  
  async submitURL(url: string, options?: SandboxSubmitOptions): Promise<string> {
    try {
      const params: any = { url };
      if (options?.timeout) {
        params.timeout = options.timeout;
      }
      
      const response = await this.client.post('/tasks/create/url', params);
      return response.data.task_id.toString();
    } catch (error: any) {
      logger.error('Cuckoo URL submission failed', { error: error.message, url });
      throw new Error(`Cuckoo URL submission failed: ${error.message}`);
    }
  }
  
  async getStatus(jobId: string): Promise<SandboxResult> {
    try {
      const response = await this.client.get(`/tasks/view/${jobId}`);
      const task = response.data.task;
      
      return {
        jobId,
        status: this.mapStatus(task.status),
        analysisId: task.id?.toString(),
      };
    } catch (error: any) {
      logger.error('Cuckoo status check failed', { error: error.message, jobId });
      throw new Error(`Cuckoo status check failed: ${error.message}`);
    }
  }
  
  async getResults(analysisId: string): Promise<SandboxResult> {
    try {
      const report = await this.client.get(`/tasks/report/${analysisId}/json`);
      const data = report.data;
      
      return {
        jobId: analysisId,
        status: 'completed',
        analysisId,
        results: {
          network: this.extractNetworkActivity(data.network),
          filesystem: this.extractFileSystemActivity(data.target?.file, data.behavior?.summary?.files),
          processes: this.extractProcessActivity(data.behavior?.processes),
          screenshots: data.screenshots?.map((s: any) => s.path || s),
          signatures: data.signatures?.map((s: any) => s.name || s),
          score: data.info?.score || 0
        }
      };
    } catch (error: any) {
      logger.error('Cuckoo results fetch failed', { error: error.message, analysisId });
      throw new Error(`Cuckoo results fetch failed: ${error.message}`);
    }
  }
  
  private mapStatus(status: string): 'pending' | 'running' | 'completed' | 'failed' {
    const mapping: Record<string, 'pending' | 'running' | 'completed' | 'failed'> = {
      'pending': 'pending',
      'running': 'running',
      'completed': 'completed',
      'reported': 'completed',
      'failed': 'failed'
    };
    return mapping[status] || 'pending';
  }
  
  private extractNetworkActivity(network: any): NetworkActivity[] {
    if (!network) return [];
    
    const activities: NetworkActivity[] = [];
    
    // Handle different Cuckoo network formats
    if (Array.isArray(network)) {
      network.forEach((item: any) => {
        activities.push({
          protocol: item.protocol || item.proto || 'tcp',
          destination: item.dst || item.destination || '',
          port: item.dport || item.port || 0,
          method: item.method,
          path: item.uri || item.path,
          statusCode: item.status_code || item.statusCode
        });
      });
    } else if (network.tcp || network.udp || network.http) {
      // Handle structured network data
      if (network.tcp) {
        network.tcp.forEach((item: any) => {
          activities.push({
            protocol: 'tcp',
            destination: item.dst || '',
            port: item.dport || 0,
          });
        });
      }
      if (network.udp) {
        network.udp.forEach((item: any) => {
          activities.push({
            protocol: 'udp',
            destination: item.dst || '',
            port: item.dport || 0,
          });
        });
      }
      if (network.http) {
        network.http.forEach((item: any) => {
          activities.push({
            protocol: 'http',
            destination: item.host || '',
            port: item.port || 80,
            method: item.method,
            path: item.uri || item.path,
            statusCode: item.status
          });
        });
      }
    }
    
    return activities;
  }
  
  private extractFileSystemActivity(file: any, filesSummary?: any): FileSystemActivity[] {
    const activities: FileSystemActivity[] = [];
    
    if (file) {
      activities.push({
        action: 'created',
        path: file.path || file.name || '',
        fileType: file.type || file.mime || ''
      });
    }
    
    if (filesSummary) {
      Object.keys(filesSummary).forEach((path: string) => {
        const actions = filesSummary[path];
        if (actions.read) {
          activities.push({ action: 'read', path });
        }
        if (actions.write || actions.created) {
          activities.push({ action: 'created', path });
        }
        if (actions.deleted) {
          activities.push({ action: 'deleted', path });
        }
      });
    }
    
    return activities;
  }
  
  private extractProcessActivity(processes: any[]): ProcessActivity[] {
    if (!processes || !Array.isArray(processes)) return [];
    
    return processes.map((proc: any) => ({
      name: proc.process_name || proc.name || '',
      pid: proc.pid || 0,
      commandLine: proc.command_line || proc.cmdline || '',
      parentPid: proc.parent_id || proc.ppid
    }));
  }
}
