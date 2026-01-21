import { DataSource, Repository } from 'typeorm';
import Redis from 'ioredis';
import { Queue, Worker } from 'bullmq';
import { BaseSandboxClient } from '../../src/integrations/base-sandbox.client';
import { mockSandboxResult } from '../fixtures/mock-data';

export const createMockDataSource = (): jest.Mocked<DataSource> => {
  const mockRepository = {
    findOne: jest.fn(),
    find: jest.fn(),
    findAndCount: jest.fn(),
    create: jest.fn(),
    save: jest.fn(),
    update: jest.fn(),
    delete: jest.fn(),
  } as unknown as jest.Mocked<Repository<any>>;

  return {
    getRepository: jest.fn().mockReturnValue(mockRepository),
    isInitialized: true,
    initialize: jest.fn().mockResolvedValue(undefined),
    destroy: jest.fn().mockResolvedValue(undefined),
  } as unknown as jest.Mocked<DataSource>;
};

export const createMockRedis = (): jest.Mocked<Redis> => {
  return {
    status: 'ready',
    get: jest.fn(),
    set: jest.fn(),
    del: jest.fn(),
    quit: jest.fn().mockResolvedValue('OK'),
  } as unknown as jest.Mocked<Redis>;
};

export const createMockSandboxClient = (): jest.Mocked<BaseSandboxClient> => {
  return {
    submitFile: jest.fn().mockResolvedValue('test-job-id'),
    submitURL: jest.fn().mockResolvedValue('test-job-id'),
    getStatus: jest.fn().mockResolvedValue({
      jobId: 'test-job-id',
      status: 'completed',
    }),
    getResults: jest.fn().mockResolvedValue(mockSandboxResult),
  } as unknown as jest.Mocked<BaseSandboxClient>;
};

export const createMockQueue = (): jest.Mocked<Queue> => {
  return {
    add: jest.fn().mockResolvedValue({} as any),
    getWaitingCount: jest.fn().mockResolvedValue(0),
    getActiveCount: jest.fn().mockResolvedValue(0),
    getCompletedCount: jest.fn().mockResolvedValue(0),
    getFailedCount: jest.fn().mockResolvedValue(0),
    close: jest.fn().mockResolvedValue(undefined),
  } as unknown as jest.Mocked<Queue>;
};

export const createMockWorker = (): jest.Mocked<Worker> => {
  return {
    on: jest.fn(),
    close: jest.fn().mockResolvedValue(undefined),
  } as unknown as jest.Mocked<Worker>;
};
