import axios, { AxiosInstance } from 'axios';
import { logger } from '../utils/logger';
import FormData from 'form-data';

export interface AnyRunSubmissionResult {
  success: boolean;
  taskId: string;
  submissionTime: Date;
}

export interface AnyRunAnalysisResult {
  taskId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  verdict: 'malicious' | 'suspicious' | 'clean' | 'unknown';
  threatLevel: number; // 0-100
  malwareFamily?: string;
  tags: string[];
  screenshot?: string;
  network: {
    connections: Array<{
      protocol: string;
      destination: string;
      port: number;
      country?: string;
    }>;
    dnsRequests: string[];
    httpRequests: string[];
  };
  processes: Array<{
    name: string;
    pid: number;
    commandLine: string;
    malicious: boolean;
  }>;
  files: Array<{
    name: string;
    hash: string;
    path: string;
    malicious: boolean;
  }>;
  behavioral: {
    registry: number;
    filesystem: number;
    network: number;
    suspicious_actions: string[];
  };
  mitre: Array<{
    tactic: string;
    technique: string;
    description: string;
  }>;
}

export class AnyRunClient {
  private client: AxiosInstance;
  private apiKey: string;
  private baseURL: string;
  private enabled: boolean;

  constructor() {
    this.apiKey = process.env.ANYRUN_API_KEY || '';
    this.baseURL = process.env.ANYRUN_URL || 'https://api.any.run/v1';
    this.enabled = !!this.apiKey;

    this.client = axios.create({
      baseURL: this.baseURL,
      headers: {
        'Authorization': `API-Key ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });

    if (!this.enabled) {
      logger.warn('Any.Run client disabled - no API key configured');
    }
  }

  /**
   * Check if Any.Run is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Test API connection
   */
  async testConnection(): Promise<boolean> {
    if (!this.enabled) return false;

    try {
      const response = await this.client.get('/limits');
      logger.info('Any.Run connection successful');
      return response.status === 200;
    } catch (error: any) {
      logger.error(`Any.Run connection failed: ${error.message}`);
      return false;
    }
  }

  /**
   * Submit URL for analysis
   */
  async submitURL(url: string, options: {
    environment?: 'windows7' | 'windows10' | 'windows11';
    timeout?: number;
    network?: 'internet' | 'isolated';
    privacy?: 'public' | 'private';
  } = {}): Promise<AnyRunSubmissionResult> {
    if (!this.enabled) {
      throw new Error('Any.Run is not configured');
    }

    try {
      const payload = {
        obj_type: 'url',
        obj_url: url,
        env_os: options.environment || 'windows10',
        env_bitness: 64,
        opt_timeout: options.timeout || 60,
        opt_network_connect: options.network === 'isolated' ? false : true,
        opt_privacy_type: options.privacy || 'private'
      };

      logger.info(`Submitting URL to Any.Run: ${url}`);

      const response = await this.client.post('/analysis', payload);

      const taskId = response.data.data.taskid;

      logger.info(`Any.Run submission successful: ${taskId}`);

      return {
        success: true,
        taskId,
        submissionTime: new Date()
      };
    } catch (error: any) {
      logger.error(`Any.Run submission failed: ${error.message}`);
      throw new Error(`Failed to submit URL to Any.Run: ${error.message}`);
    }
  }

  /**
   * Submit file for analysis
   */
  async submitFile(fileBuffer: Buffer, fileName: string, options: {
    environment?: 'windows7' | 'windows10' | 'windows11';
    timeout?: number;
    network?: 'internet' | 'isolated';
    privacy?: 'public' | 'private';
  } = {}): Promise<AnyRunSubmissionResult> {
    if (!this.enabled) {
      throw new Error('Any.Run is not configured');
    }

    try {
      const formData = new FormData();
      formData.append('file', fileBuffer, fileName);
      formData.append('env_os', options.environment || 'windows10');
      formData.append('env_bitness', '64');
      formData.append('opt_timeout', String(options.timeout || 60));
      formData.append('opt_network_connect', options.network === 'isolated' ? 'false' : 'true');
      formData.append('opt_privacy_type', options.privacy || 'private');

      logger.info(`Submitting file to Any.Run: ${fileName}`);

      const response = await this.client.post('/analysis', formData, {
        headers: {
          ...formData.getHeaders(),
          'Authorization': `API-Key ${this.apiKey}`
        }
      });

      const taskId = response.data.data.taskid;

      logger.info(`Any.Run file submission successful: ${taskId}`);

      return {
        success: true,
        taskId,
        submissionTime: new Date()
      };
    } catch (error: any) {
      logger.error(`Any.Run file submission failed: ${error.message}`);
      throw new Error(`Failed to submit file to Any.Run: ${error.message}`);
    }
  }

  /**
   * Get analysis status and results
   */
  async getAnalysis(taskId: string): Promise<AnyRunAnalysisResult> {
    if (!this.enabled) {
      throw new Error('Any.Run is not configured');
    }

    try {
      const response = await this.client.get(`/analysis/${taskId}`);
      const data = response.data.data;

      // Map Any.Run status to our status
      let status: 'pending' | 'running' | 'completed' | 'failed';
      switch (data.status) {
        case 'created':
        case 'in_queue':
          status = 'pending';
          break;
        case 'in_progress':
          status = 'running';
          break;
        case 'done':
          status = 'completed';
          break;
        case 'failed':
        case 'error':
          status = 'failed';
          break;
        default:
          status = 'pending';
      }

      // Map verdict
      let verdict: 'malicious' | 'suspicious' | 'clean' | 'unknown' = 'unknown';
      if (data.verdict) {
        if (data.verdict.threat_level >= 70) verdict = 'malicious';
        else if (data.verdict.threat_level >= 40) verdict = 'suspicious';
        else if (data.verdict.threat_level < 40) verdict = 'clean';
      }

      return {
        taskId,
        status,
        verdict,
        threatLevel: data.verdict?.threat_level || 0,
        malwareFamily: data.verdict?.malware_family,
        tags: data.tags || [],
        screenshot: data.screenshots?.[0],
        network: {
          connections: (data.network?.connections || []).map((conn: any) => ({
            protocol: conn.protocol,
            destination: conn.ip,
            port: conn.port,
            country: conn.country
          })),
          dnsRequests: data.network?.dns || [],
          httpRequests: data.network?.http || []
        },
        processes: (data.processes || []).map((proc: any) => ({
          name: proc.name,
          pid: proc.pid,
          commandLine: proc.cmd,
          malicious: proc.malicious || false
        })),
        files: (data.files || []).map((file: any) => ({
          name: file.name,
          hash: file.hash,
          path: file.path,
          malicious: file.malicious || false
        })),
        behavioral: {
          registry: data.behavior?.registry_changes || 0,
          filesystem: data.behavior?.file_changes || 0,
          network: data.behavior?.network_events || 0,
          suspicious_actions: data.behavior?.suspicious || []
        },
        mitre: (data.mitre_attack || []).map((mitre: any) => ({
          tactic: mitre.tactic,
          technique: mitre.technique,
          description: mitre.description
        }))
      };
    } catch (error: any) {
      logger.error(`Failed to get Any.Run analysis: ${error.message}`);
      throw new Error(`Failed to get analysis from Any.Run: ${error.message}`);
    }
  }

  /**
   * Wait for analysis to complete with polling
   */
  async waitForAnalysis(
    taskId: string,
    maxWaitTime: number = 300000, // 5 minutes
    pollInterval: number = 10000   // 10 seconds
  ): Promise<AnyRunAnalysisResult> {
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitTime) {
      const result = await this.getAnalysis(taskId);

      if (result.status === 'completed' || result.status === 'failed') {
        return result;
      }

      logger.debug(`Any.Run task ${taskId} still ${result.status}, waiting...`);
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Analysis timeout after ${maxWaitTime}ms`);
  }

  /**
   * Get account limits
   */
  async getLimits(): Promise<{
    submissions_remaining: number;
    submissions_total: number;
    reset_date: Date;
  }> {
    if (!this.enabled) {
      throw new Error('Any.Run is not configured');
    }

    try {
      const response = await this.client.get('/limits');
      const data = response.data.data;

      return {
        submissions_remaining: data.submissions.remaining,
        submissions_total: data.submissions.total,
        reset_date: new Date(data.submissions.reset_at)
      };
    } catch (error: any) {
      logger.error(`Failed to get Any.Run limits: ${error.message}`);
      throw error;
    }
  }

  /**
   * Search for previous analyses
   */
  async searchURL(url: string): Promise<AnyRunAnalysisResult[]> {
    if (!this.enabled) {
      throw new Error('Any.Run is not configured');
    }

    try {
      const response = await this.client.get('/analysis/search', {
        params: {
          obj_url: url,
          limit: 10
        }
      });

      const analyses = response.data.data.tasks || [];

      return Promise.all(
        analyses.map((task: any) => this.getAnalysis(task.taskid))
      );
    } catch (error: any) {
      logger.error(`Failed to search Any.Run: ${error.message}`);
      return [];
    }
  }
}
