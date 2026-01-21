import { SandboxQueueJob } from '../../../src/jobs/sandbox-queue.job';
import { ResultProcessorService } from '../../../src/services/result-processor.service';
import Redis from 'ioredis';
import { Queue, Worker, QueueEvents } from 'bullmq';
import { createMockRedis } from '../../helpers/mocks';

jest.mock('bullmq');

describe('SandboxQueueJob', () => {
  let queueJob: SandboxQueueJob;
  let mockRedis: jest.Mocked<Redis>;
  let mockResultProcessor: jest.Mocked<ResultProcessorService>;
  let mockQueue: jest.Mocked<Queue>;
  let mockWorker: jest.Mocked<Worker>;
  let mockQueueEvents: jest.Mocked<QueueEvents>;

  beforeEach(() => {
    jest.clearAllMocks();

    mockRedis = createMockRedis();
    mockResultProcessor = {
      processResults: jest.fn().mockResolvedValue(undefined),
    } as any;

    mockQueue = {
      add: jest.fn().mockResolvedValue({} as any),
      getWaitingCount: jest.fn().mockResolvedValue(0),
      getActiveCount: jest.fn().mockResolvedValue(0),
      getCompletedCount: jest.fn().mockResolvedValue(10),
      getFailedCount: jest.fn().mockResolvedValue(1),
      close: jest.fn().mockResolvedValue(undefined),
    } as any;

    mockWorker = {
      on: jest.fn(),
      close: jest.fn().mockResolvedValue(undefined),
    } as any;

    mockQueueEvents = {
      on: jest.fn(),
      close: jest.fn().mockResolvedValue(undefined),
    } as any;

    (Queue as jest.MockedClass<typeof Queue>).mockImplementation(() => mockQueue);
    (Worker as jest.MockedClass<typeof Worker>).mockImplementation(() => mockWorker);
    (QueueEvents as jest.MockedClass<typeof QueueEvents>).mockImplementation(() => mockQueueEvents);

    queueJob = new SandboxQueueJob(mockRedis, mockResultProcessor);
  });

  describe('addAnalysisJob', () => {
    it('should add job to queue', async () => {
      await queueJob.addAnalysisJob('test-analysis-id', 30);

      expect(mockQueue.add).toHaveBeenCalledWith(
        'process-analysis',
        { analysisId: 'test-analysis-id' },
        expect.objectContaining({
          delay: 30000,
          attempts: 3,
          backoff: expect.any(Object),
        })
      );
    });

    it('should add job without delay', async () => {
      await queueJob.addAnalysisJob('test-analysis-id', 0);

      expect(mockQueue.add).toHaveBeenCalledWith(
        'process-analysis',
        { analysisId: 'test-analysis-id' },
        expect.objectContaining({
          delay: 0,
        })
      );
    });
  });

  describe('getQueueStats', () => {
    it('should return queue statistics', async () => {
      const stats = await queueJob.getQueueStats();

      expect(stats).toEqual({
        waiting: 0,
        active: 0,
        completed: 10,
        failed: 1,
      });
    });
  });

  describe('close', () => {
    it('should close queue and worker', async () => {
      await queueJob.close();

      expect(mockWorker.close).toHaveBeenCalled();
      expect(mockQueueEvents.close).toHaveBeenCalled();
      expect(mockQueue.close).toHaveBeenCalled();
    });
  });

  describe('event handlers', () => {
    it('should set up event handlers', () => {
      expect(mockWorker.on).toHaveBeenCalledWith('completed', expect.any(Function));
      expect(mockWorker.on).toHaveBeenCalledWith('failed', expect.any(Function));
      expect(mockWorker.on).toHaveBeenCalledWith('error', expect.any(Function));
    });
  });
});
