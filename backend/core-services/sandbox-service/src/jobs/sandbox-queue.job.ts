import { Queue, Worker, QueueEvents } from 'bullmq';
import Redis from 'ioredis';
import { ResultProcessorService } from '../services/result-processor.service';
import { logger } from '../utils/logger';
import { config } from '../config';
import { QUEUE_NAMES } from '../../../../shared/database/redis/queue-keys';

export interface SandboxJobData {
  analysisId: string;
  jobId: string;
  retryCount?: number;
}

export class SandboxQueueJob {
  private queue: Queue;
  private worker: Worker;
  private queueEvents: QueueEvents;
  private resultProcessor: ResultProcessorService;
  private redis: Redis;
  
  constructor(redis: Redis, resultProcessor: ResultProcessorService) {
    this.redis = redis;
    this.resultProcessor = resultProcessor;
    
    // Create queue
    this.queue = new Queue(QUEUE_NAMES.SANDBOX_JOBS, {
      connection: redis as any,
    });
    
    // Create worker
    this.worker = new Worker(
      QUEUE_NAMES.SANDBOX_JOBS,
      async (job) => {
        const { analysisId } = job.data as SandboxJobData;
        await this.resultProcessor.processResults(analysisId);
      },
      {
        connection: redis as any,
        concurrency: config.queue.concurrency,
        removeOnComplete: {
          age: 3600, // Keep completed jobs for 1 hour
          count: 1000, // Keep max 1000 completed jobs
        },
        removeOnFail: {
          age: 86400, // Keep failed jobs for 24 hours
        },
      }
    );
    
    // Create queue events listener
    this.queueEvents = new QueueEvents(QUEUE_NAMES.SANDBOX_JOBS, {
      connection: redis as any,
    });
    
    this.setupEventHandlers();
  }
  
  private setupEventHandlers(): void {
    this.worker.on('completed', (job) => {
      logger.info(`Sandbox analysis job completed: ${job.id}`, {
        analysisId: job.data.analysisId
      });
    });
    
    this.worker.on('failed', (job, err) => {
      logger.error(`Sandbox analysis job failed: ${job?.id}`, {
        error: err.message,
        analysisId: job?.data?.analysisId,
        stack: err.stack
      });
    });
    
    this.worker.on('error', (err) => {
      logger.error('Sandbox queue worker error', err);
    });
    
    this.queueEvents.on('completed', ({ jobId }) => {
      logger.debug(`Queue event: job ${jobId} completed`);
    });
    
    this.queueEvents.on('failed', ({ jobId, failedReason }) => {
      logger.warn(`Queue event: job ${jobId} failed`, { reason: failedReason });
    });
  }
  
  /**
   * Add analysis job to queue
   */
  async addAnalysisJob(analysisId: string, delay: number = 0): Promise<void> {
    await this.queue.add(
      'process-analysis',
      {
        analysisId,
      } as SandboxJobData,
      {
        delay: delay * 1000, // Delay in milliseconds
        attempts: 3,
        backoff: {
          type: 'exponential',
          delay: 5000, // Start with 5 seconds
        },
        removeOnComplete: {
          age: 3600,
          count: 1000,
        },
        removeOnFail: {
          age: 86400,
        },
      }
    );
    
    logger.info(`Added sandbox analysis job to queue`, { analysisId, delay });
  }
  
  /**
   * Get queue statistics
   */
  async getQueueStats(): Promise<{
    waiting: number;
    active: number;
    completed: number;
    failed: number;
  }> {
    const [waiting, active, completed, failed] = await Promise.all([
      this.queue.getWaitingCount(),
      this.queue.getActiveCount(),
      this.queue.getCompletedCount(),
      this.queue.getFailedCount(),
    ]);
    
    return {
      waiting,
      active,
      completed,
      failed,
    };
  }
  
  /**
   * Close queue and worker
   */
  async close(): Promise<void> {
    await this.worker.close();
    await this.queueEvents.close();
    await this.queue.close();
    logger.info('Sandbox queue closed');
  }
}
