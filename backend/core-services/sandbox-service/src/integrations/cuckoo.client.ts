import axios, { AxiosInstance } from 'axios';
import { logger } from '../utils/logger';
import FormData from 'form-data';

export interface CuckooSubmissionResult {
  success: boolean;
  taskId: number;
  submissionTime: Date;
}

export interface CuckooAnalysisResult {
  taskId: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  verdict: 'malicious' | 'suspicious' | 'clean' | 'unknown';
  score: number; // 0-10
  malwareFamily?: string;
  tags: string[];
  signatures: Array<{
    name: string;
    description: string;
    severity: number;
    marks: number;
  }>;
  network: {
    hosts: string[];
    domains: string[];
    http: Array<{
      method: string;
      host: string;
      uri: string;
      status: number;
    }>;
    dns: Array<{
      request: string;
      answers: string[];
    }>;
    tcp: Array<{
      src: string;
      dst: string;
      dport: number;
    }>;
    udp: Array<{
      src: string;
      dst: string;
      dport: number;
    }>;
  };
  processes: Array<{
    process_name: string;
    process_id: number;
    parent_id: number;
    command_line: string;
    first_seen: number;
  }>;
  files: Array<{
    name: string;
    path: string;
    md5: string;
    sha1: string;
    sha256: string;
    type: string;
  }>;
  registry: Array<{
    key: string;
    value: string;
    data: string;
  }>;
  dropped: Array<{
    name: string;
    path: string;
    size: number;
    md5: string;
    type: string;
  }>;
  screenshots: string[];
  mitre: Array<{
    id: string;
    tactic: string;
    technique: string;
    description: string;
  }>;
}

export class CuckooClient {
  private client: AxiosInstance;
  private apiToken: string;
  private baseURL: string;
  private enabled: boolean;

  constructor() {
    this.apiToken = process.env.CUCKOO_API_TOKEN || '';
    this.baseURL = process.env.CUCKOO_URL || 'http://localhost:8090';
    this.enabled = !!this.apiToken;

    this.client = axios.create({
      baseURL: this.baseURL,
      headers: {
        'Authorization': `Bearer ${this.apiToken}`
      },
      timeout: 30000
    });

    if (!this.enabled) {
      logger.warn('Cuckoo Sandbox client disabled - no API token configured');
    }
  }

  /**
   * Check if Cuckoo is enabled
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
      const response = await this.client.get('/cuckoo/status');
      logger.info('Cuckoo Sandbox connection successful');
      return response.status === 200;
    } catch (error: any) {
      logger.error(`Cuckoo connection failed: ${error.message}`);
      return false;
    }
  }

  /**
   * Submit URL for analysis
   */
  async submitURL(url: string, options: {
    package?: string;
    timeout?: number;
    priority?: number;
    tags?: string[];
    memory?: boolean;
    enforce_timeout?: boolean;
  } = {}): Promise<CuckooSubmissionResult> {
    if (!this.enabled) {
      throw new Error('Cuckoo Sandbox is not configured');
    }

    try {
      const formData = new FormData();
      formData.append('url', url);
      
      if (options.package) formData.append('package', options.package);
      if (options.timeout) formData.append('timeout', String(options.timeout));
      if (options.priority) formData.append('priority', String(options.priority));
      if (options.tags) formData.append('tags', options.tags.join(','));
      if (options.memory !== undefined) formData.append('memory', String(options.memory));
      if (options.enforce_timeout !== undefined) formData.append('enforce_timeout', String(options.enforce_timeout));

      logger.info(`Submitting URL to Cuckoo: ${url}`);

      const response = await this.client.post('/tasks/create/url', formData, {
        headers: formData.getHeaders()
      });

      const taskId = response.data.task_id;

      logger.info(`Cuckoo submission successful: ${taskId}`);

      return {
        success: true,
        taskId,
        submissionTime: new Date()
      };
    } catch (error: any) {
      logger.error(`Cuckoo submission failed: ${error.message}`);
      throw new Error(`Failed to submit URL to Cuckoo: ${error.message}`);
    }
  }

  /**
   * Submit file for analysis
   */
  async submitFile(fileBuffer: Buffer, fileName: string, options: {
    package?: string;
    timeout?: number;
    priority?: number;
    tags?: string[];
    memory?: boolean;
    enforce_timeout?: boolean;
  } = {}): Promise<CuckooSubmissionResult> {
    if (!this.enabled) {
      throw new Error('Cuckoo Sandbox is not configured');
    }

    try {
      const formData = new FormData();
      formData.append('file', fileBuffer, fileName);
      
      if (options.package) formData.append('package', options.package);
      if (options.timeout) formData.append('timeout', String(options.timeout));
      if (options.priority) formData.append('priority', String(options.priority));
      if (options.tags) formData.append('tags', options.tags.join(','));
      if (options.memory !== undefined) formData.append('memory', String(options.memory));
      if (options.enforce_timeout !== undefined) formData.append('enforce_timeout', String(options.enforce_timeout));

      logger.info(`Submitting file to Cuckoo: ${fileName}`);

      const response = await this.client.post('/tasks/create/file', formData, {
        headers: formData.getHeaders()
      });

      const taskId = response.data.task_id;

      logger.info(`Cuckoo file submission successful: ${taskId}`);

      return {
        success: true,
        taskId,
        submissionTime: new Date()
      };
    } catch (error: any) {
      logger.error(`Cuckoo file submission failed: ${error.message}`);
      throw new Error(`Failed to submit file to Cuckoo: ${error.message}`);
    }
  }

  /**
   * Get task status
   */
  async getTaskStatus(taskId: number): Promise<{
    status: 'pending' | 'running' | 'completed' | 'failed';
    target: string;
  }> {
    if (!this.enabled) {
      throw new Error('Cuckoo Sandbox is not configured');
    }

    try {
      const response = await this.client.get(`/tasks/view/${taskId}`);
      const task = response.data.task;

      let status: 'pending' | 'running' | 'completed' | 'failed';
      switch (task.status) {
        case 'pending':
          status = 'pending';
          break;
        case 'running':
          status = 'running';
          break;
        case 'completed':
        case 'reported':
          status = 'completed';
          break;
        case 'failed_analysis':
        case 'failed_processing':
        case 'failed_reporting':
          status = 'failed';
          break;
        default:
          status = 'pending';
      }

      return {
        status,
        target: task.target
      };
    } catch (error: any) {
      logger.error(`Failed to get Cuckoo task status: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get full analysis report
   */
  async getReport(taskId: number): Promise<CuckooAnalysisResult> {
    if (!this.enabled) {
      throw new Error('Cuckoo Sandbox is not configured');
    }

    try {
      const response = await this.client.get(`/tasks/report/${taskId}`);
      const report = response.data;

      // Calculate verdict based on score and signatures
      let verdict: 'malicious' | 'suspicious' | 'clean' | 'unknown' = 'unknown';
      const score = report.info?.score || 0;
      
      if (score >= 7) verdict = 'malicious';
      else if (score >= 4) verdict = 'suspicious';
      else if (score < 4) verdict = 'clean';

      // Extract malware family from signatures
      const malwareFamily = report.malfamily || 
        report.signatures?.find((sig: any) => sig.name.includes('family'))?.name;

      // Extract MITRE ATT&CK techniques
      const mitre = [];
      if (report.signatures) {
        for (const sig of report.signatures) {
          if (sig.mitre_attack) {
            mitre.push({
              id: sig.mitre_attack.id,
              tactic: sig.mitre_attack.tactic,
              technique: sig.mitre_attack.technique,
              description: sig.description
            });
          }
        }
      }

      return {
        taskId,
        status: 'completed',
        verdict,
        score,
        malwareFamily,
        tags: report.info?.tags || [],
        signatures: (report.signatures || []).map((sig: any) => ({
          name: sig.name,
          description: sig.description,
          severity: sig.severity,
          marks: sig.marks?.length || 0
        })),
        network: {
          hosts: report.network?.hosts || [],
          domains: report.network?.domains || [],
          http: report.network?.http || [],
          dns: report.network?.dns || [],
          tcp: report.network?.tcp || [],
          udp: report.network?.udp || []
        },
        processes: report.behavior?.processes || [],
        files: (report.dropped || []).map((file: any) => ({
          name: file.name,
          path: file.path,
          md5: file.md5,
          sha1: file.sha1,
          sha256: file.sha256,
          type: file.type
        })),
        registry: report.behavior?.summary?.keys || [],
        dropped: report.dropped || [],
        screenshots: (report.screenshots || []).map((s: any) => s.path),
        mitre
      };
    } catch (error: any) {
      logger.error(`Failed to get Cuckoo report: ${error.message}`);
      throw new Error(`Failed to get report from Cuckoo: ${error.message}`);
    }
  }

  /**
   * Wait for analysis to complete with polling
   */
  async waitForAnalysis(
    taskId: number,
    maxWaitTime: number = 300000, // 5 minutes
    pollInterval: number = 10000   // 10 seconds
  ): Promise<CuckooAnalysisResult> {
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitTime) {
      const status = await this.getTaskStatus(taskId);

      if (status.status === 'completed') {
        return await this.getReport(taskId);
      }

      if (status.status === 'failed') {
        throw new Error(`Analysis failed for task ${taskId}`);
      }

      logger.debug(`Cuckoo task ${taskId} still ${status.status}, waiting...`);
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error(`Analysis timeout after ${maxWaitTime}ms`);
  }

  /**
   * Delete task and its data
   */
  async deleteTask(taskId: number): Promise<boolean> {
    if (!this.enabled) {
      throw new Error('Cuckoo Sandbox is not configured');
    }

    try {
      await this.client.get(`/tasks/delete/${taskId}`);
      logger.info(`Deleted Cuckoo task: ${taskId}`);
      return true;
    } catch (error: any) {
      logger.error(`Failed to delete Cuckoo task: ${error.message}`);
      return false;
    }
  }

  /**
   * Get Cuckoo status and stats
   */
  async getStatus(): Promise<{
    version: string;
    tasks: {
      total: number;
      pending: number;
      running: number;
      completed: number;
      failed: number;
    };
  }> {
    if (!this.enabled) {
      throw new Error('Cuckoo Sandbox is not configured');
    }

    try {
      const statusResponse = await this.client.get('/cuckoo/status');
      const tasksResponse = await this.client.get('/tasks/list');

      const tasks = tasksResponse.data.tasks || [];

      return {
        version: statusResponse.data.version,
        tasks: {
          total: tasks.length,
          pending: tasks.filter((t: any) => t.status === 'pending').length,
          running: tasks.filter((t: any) => t.status === 'running').length,
          completed: tasks.filter((t: any) => t.status === 'reported').length,
          failed: tasks.filter((t: any) => t.status.includes('failed')).length
        }
      };
    } catch (error: any) {
      logger.error(`Failed to get Cuckoo status: ${error.message}`);
      throw error;
    }
  }

  /**
   * Search for tasks by target
   */
  async searchTasks(target: string): Promise<number[]> {
    if (!this.enabled) {
      throw new Error('Cuckoo Sandbox is not configured');
    }

    try {
      const response = await this.client.get('/tasks/list');
      const tasks = response.data.tasks || [];

      return tasks
        .filter((t: any) => t.target.includes(target))
        .map((t: any) => t.id);
    } catch (error: any) {
      logger.error(`Failed to search Cuckoo tasks: ${error.message}`);
      return [];
    }
  }
}
