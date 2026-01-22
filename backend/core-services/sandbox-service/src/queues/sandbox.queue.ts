import { Queue, Worker, Job, QueueEvents } from 'bullmq';
import { Redis } from 'ioredis';
import { logger } from '../utils/logger';
import { AnyRunClient } from '../integrations/anyrun.client';
import { CuckooClient } from '../integrations/cuckoo.client';
import { getPostgreSQL } from '../../../../shared/database/connection';
import { SandboxAnalysis } from '../../../../shared/database/models/SandboxAnalysis';

export interface SandboxJobData {
  analysisId: string;
  url?: string;
  fileBuffer?: Buffer;
  fileName?: string;
  organizationId: string;
  provider: 'anyrun' | 'cuckoo' | 'auto';
  priority: number;
  options?: {
    timeout?: number;
    environment?: string;
    network?: 'internet' | 'isolated';
  };
}

export interface SandboxJobResult {
  success: boolean;
  verdict: 'malicious' | 'suspicious' | 'clean' | 'unknown';
  score: number;
  taskId: string | number;
  provider: string;
  analysis: any;
}

export class SandboxQueue {
  private queue: Queue<SandboxJobData, SandboxJobResult>;
  private worker: Worker<SandboxJobData, SandboxJobResult>;
  private queueEvents: QueueEvents;
  private redisConnection: Redis;
  private anyrunClient: AnyRunClient;
  private cuckooClient: CuckooClient;

  constructor() {
    // Create Redis connection
    const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
    this.redisConnection = new Redis(redisUrl, {
      maxRetriesPerRequest: null,
      enableReadyCheck: false
    });

    // Initialize clients
    this.anyrunClient = new AnyRunClient();
    this.cuckooClient = new CuckooClient();

    // Create queue
    this.queue = new Queue<SandboxJobData, SandboxJobResult>('sandbox-analysis', {
      connection: this.redisConnection,
      defaultJobOptions: {
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 5000
        },
        removeOnComplete: {
          age: 86400, // Keep for 24 hours
          count: 1000
        },
        removeOnFail: {
          age: 604800 // Keep failures for 7 days
        }
      }
    });

    // Create worker
    this.worker = new Worker<SandboxJobData, SandboxJobResult>(
      'sandbox-analysis',
      async (job: Job<SandboxJobData>) => this.processJob(job),
      {
        connection: this.redisConnection,
        concurrency: 5, // Process 5 jobs concurrently
        limiter: {
          max: 10, // Max 10 jobs
          duration: 60000 // per minute
        }
      }
    );

    // Create queue events listener
    this.queueEvents = new QueueEvents('sandbox-analysis', {
      connection: this.redisConnection
    });

    this.setupEventListeners();

    logger.info('Sandbox queue initialized');
  }

  /**
   * Setup event listeners
   */
  private setupEventListeners(): void {
    // Worker events
    this.worker.on('completed', (job: Job, result: SandboxJobResult) => {
      logger.info(`Sandbox job ${job.id} completed: ${result.verdict} (score: ${result.score})`);
    });

    this.worker.on('failed', (job: Job | undefined, error: Error) => {
      logger.error(`Sandbox job ${job?.id} failed: ${error.message}`);
    });

    this.worker.on('error', (error: Error) => {
      logger.error(`Sandbox worker error: ${error.message}`);
    });

    // Queue events
    this.queueEvents.on('waiting', ({ jobId }) => {
      logger.debug(`Sandbox job ${jobId} is waiting`);
    });

    this.queueEvents.on('active', ({ jobId }) => {
      logger.debug(`Sandbox job ${jobId} is active`);
    });

    this.queueEvents.on('stalled', ({ jobId }) => {
      logger.warn(`Sandbox job ${jobId} stalled`);
    });

    this.queueEvents.on('progress', ({ jobId, data }) => {
      logger.debug(`Sandbox job ${jobId} progress: ${JSON.stringify(data)}`);
    });
  }

  /**
   * Add URL analysis job to queue
   */
  async addURLAnalysis(data: Omit<SandboxJobData, 'fileBuffer' | 'fileName'>): Promise<string> {
    const job = await this.queue.add('analyze-url', data, {
      priority: data.priority || 1,
      jobId: data.analysisId
    });

    logger.info(`Added URL analysis job: ${job.id} (${data.url})`);

    return job.id!;
  }

  /**
   * Add file analysis job to queue
   */
  async addFileAnalysis(data: Omit<SandboxJobData, 'url'>): Promise<string> {
    const job = await this.queue.add('analyze-file', data, {
      priority: data.priority || 1,
      jobId: data.analysisId
    });

    logger.info(`Added file analysis job: ${job.id} (${data.fileName})`);

    return job.id!;
  }

  /**
   * Process sandbox job
   */
  private async processJob(job: Job<SandboxJobData>): Promise<SandboxJobResult> {
    logger.info(`Processing sandbox job: ${job.id}`);

    const { analysisId, url, fileBuffer, fileName, provider, options } = job.data;

    // Update job progress
    await job.updateProgress({ status: 'selecting_provider' });

    // Determine which provider to use
    const selectedProvider = await this.selectProvider(provider);

    if (!selectedProvider) {
      throw new Error('No sandbox provider available');
    }

    await job.updateProgress({ status: 'submitting', provider: selectedProvider });

    try {
      let result: SandboxJobResult;

      if (selectedProvider === 'anyrun') {
        result = await this.processWithAnyRun(job);
      } else if (selectedProvider === 'cuckoo') {
        result = await this.processWithCuckoo(job);
      } else {
        throw new Error(`Unknown provider: ${selectedProvider}`);
      }

      // Update database
      await this.updateAnalysisInDatabase(analysisId, result);

      return result;

    } catch (error: any) {
      logger.error(`Sandbox job ${job.id} failed: ${error.message}`);
      
      // Update database with error
      await this.updateAnalysisInDatabase(analysisId, {
        success: false,
        verdict: 'unknown',
        score: 0,
        taskId: '',
        provider: selectedProvider,
        analysis: { error: error.message }
      });

      throw error;
    }
  }

  /**
   * Process with Any.Run
   */
  private async processWithAnyRun(job: Job<SandboxJobData>): Promise<SandboxJobResult> {
    const { url, fileBuffer, fileName, options } = job.data;

    // Submit to Any.Run
    await job.updateProgress({ status: 'submitting_anyrun' });

    let submission;
    if (url) {
      submission = await this.anyrunClient.submitURL(url, {
        timeout: options?.timeout || 60,
        network: options?.network || 'internet'
      });
    } else if (fileBuffer && fileName) {
      submission = await this.anyrunClient.submitFile(fileBuffer, fileName, {
        timeout: options?.timeout || 60,
        network: options?.network || 'internet'
      });
    } else {
      throw new Error('Either URL or file must be provided');
    }

    // Wait for analysis
    await job.updateProgress({ status: 'analyzing', taskId: submission.taskId });

    const analysis = await this.anyrunClient.waitForAnalysis(
      submission.taskId,
      300000, // 5 minutes max
      15000   // Check every 15 seconds
    );

    return {
      success: true,
      verdict: analysis.verdict,
      score: analysis.threatLevel / 10, // Convert 0-100 to 0-10
      taskId: submission.taskId,
      provider: 'anyrun',
      analysis: analysis
    };
  }

  /**
   * Process with Cuckoo
   */
  private async processWithCuckoo(job: Job<SandboxJobData>): Promise<SandboxJobResult> {
    const { url, fileBuffer, fileName, options } = job.data;

    // Submit to Cuckoo
    await job.updateProgress({ status: 'submitting_cuckoo' });

    let submission;
    if (url) {
      submission = await this.cuckooClient.submitURL(url, {
        timeout: options?.timeout || 60,
        priority: job.data.priority
      });
    } else if (fileBuffer && fileName) {
      submission = await this.cuckooClient.submitFile(fileBuffer, fileName, {
        timeout: options?.timeout || 60,
        priority: job.data.priority
      });
    } else {
      throw new Error('Either URL or file must be provided');
    }

    // Wait for analysis
    await job.updateProgress({ status: 'analyzing', taskId: submission.taskId });

    const analysis = await this.cuckooClient.waitForAnalysis(
      submission.taskId,
      300000, // 5 minutes max
      15000   // Check every 15 seconds
    );

    return {
      success: true,
      verdict: analysis.verdict,
      score: analysis.score,
      taskId: submission.taskId.toString(),
      provider: 'cuckoo',
      analysis: analysis
    };
  }

  /**
   * Select appropriate provider
   */
  private async selectProvider(requested: 'anyrun' | 'cuckoo' | 'auto'): Promise<'anyrun' | 'cuckoo' | null> {
    if (requested === 'anyrun' && this.anyrunClient.isEnabled()) {
      return 'anyrun';
    }

    if (requested === 'cuckoo' && this.cuckooClient.isEnabled()) {
      return 'cuckoo';
    }

    if (requested === 'auto') {
      // Prefer Any.Run if available
      if (this.anyrunClient.isEnabled()) {
        return 'anyrun';
      }
      if (this.cuckooClient.isEnabled()) {
        return 'cuckoo';
      }
    }

    return null;
  }

  /**
   * Update analysis in database
   */
  private async updateAnalysisInDatabase(analysisId: string, result: SandboxJobResult): Promise<void> {
    try {
      const dataSource = getPostgreSQL();
      const repository = dataSource.getRepository(SandboxAnalysis);

      await repository.update(analysisId, {
        status: result.success ? 'completed' : 'failed',
        submission_id: result.taskId.toString(),
        sandbox_provider: result.provider,
        analysis_result: result.analysis,
        risk_score: result.score,
        malware_family: result.analysis.malwareFamily,
        behavioral_indicators: result.analysis.behavioral || {},
        network_connections: result.analysis.network || {},
        processes: result.analysis.processes || [],
        completed_at: new Date(),
        error_message: result.success ? null : (result.analysis?.error || 'Analysis failed')
      });

      logger.info(`Updated sandbox analysis in database: ${analysisId}`);
    } catch (error: any) {
      logger.error(`Failed to update database: ${error.message}`);
    }
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId: string): Promise<{
    state: string;
    progress: any;
    result?: SandboxJobResult;
    error?: string;
  }> {
    const job = await this.queue.getJob(jobId);

    if (!job) {
      throw new Error(`Job ${jobId} not found`);
    }

    const state = await job.getState();
    const progress = job.progress;

    let result;
    if (state === 'completed') {
      result = job.returnvalue;
    }

    let error;
    if (state === 'failed') {
      error = job.failedReason;
    }

    return { state, progress, result, error };
  }

  /**
   * Get queue statistics
   */
  async getStats(): Promise<{
    waiting: number;
    active: number;
    completed: number;
    failed: number;
    delayed: number;
  }> {
    const [waiting, active, completed, failed, delayed] = await Promise.all([
      this.queue.getWaitingCount(),
      this.queue.getActiveCount(),
      this.queue.getCompletedCount(),
      this.queue.getFailedCount(),
      this.queue.getDelayedCount()
    ]);

    return { waiting, active, completed, failed, delayed };
  }

  /**
   * Pause queue
   */
  async pause(): Promise<void> {
    await this.queue.pause();
    logger.info('Sandbox queue paused');
  }

  /**
   * Resume queue
   */
  async resume(): Promise<void> {
    await this.queue.resume();
    logger.info('Sandbox queue resumed');
  }

  /**
   * Close queue and worker
   */
  async close(): Promise<void> {
    await this.worker.close();
    await this.queue.close();
    await this.queueEvents.close();
    await this.redisConnection.quit();
    logger.info('Sandbox queue closed');
  }
}
