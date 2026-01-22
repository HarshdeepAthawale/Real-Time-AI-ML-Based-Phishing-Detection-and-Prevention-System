import 'reflect-metadata';
import { ResultProcessorService } from '../../../src/services/result-processor.service';
import { BehavioralAnalyzerService } from '../../../src/services/behavioral-analyzer.service';
import { DataSource, Repository } from 'typeorm';
import { SandboxAnalysis } from '../../../../shared/database/models/SandboxAnalysis';
import { Threat } from '../../../../shared/database/models/Threat';
import { createMockSandboxClient } from '../../helpers/mocks';
import {
  mockSandboxResult,
  mockBehavioralAnalysis,
  mockSandboxAnalysisRecord,
  mockThreatRecord,
} from '../../fixtures/mock-data';

describe('ResultProcessorService', () => {
  let service: ResultProcessorService;
  let mockSandboxClient: any;
  let mockBehavioralAnalyzer: jest.Mocked<BehavioralAnalyzerService>;
  let mockDataSource: jest.Mocked<DataSource>;
  let mockSandboxRepository: jest.Mocked<Repository<any>>;
  let mockThreatRepository: jest.Mocked<Repository<any>>;

  beforeEach(() => {
    mockSandboxClient = createMockSandboxClient();
    mockSandboxClient.getResults.mockResolvedValue(mockSandboxResult);

    mockBehavioralAnalyzer = {
      analyze: jest.fn().mockReturnValue(mockBehavioralAnalysis),
    } as any;

    mockSandboxRepository = {
      findOne: jest.fn().mockResolvedValue({
        ...mockSandboxAnalysisRecord,
        sandbox_job_id: 'test-job-id',
      }),
      save: jest.fn().mockResolvedValue(mockSandboxAnalysisRecord),
    } as any;

    mockThreatRepository = {
      create: jest.fn().mockReturnValue(mockThreatRecord),
      save: jest.fn().mockResolvedValue(mockThreatRecord),
    } as any;

    mockDataSource = {
      getRepository: jest.fn((entity: any) => {
        if (entity === SandboxAnalysis) {
          return mockSandboxRepository;
        }
        if (entity === Threat) {
          return mockThreatRepository;
        }
        return mockSandboxRepository;
      }),
    } as any;

    service = new ResultProcessorService(
      mockSandboxClient,
      mockDataSource,
      mockBehavioralAnalyzer
    );
  });

  describe('processResults', () => {
    it('should process completed analysis successfully', async () => {
      await service.processResults('test-analysis-id');

      expect(mockSandboxClient.getResults).toHaveBeenCalledWith('test-job-id');
      expect(mockBehavioralAnalyzer.analyze).toHaveBeenCalled();
      expect(mockSandboxRepository.save).toHaveBeenCalled();
    });

    it('should create threat record for malicious analysis', async () => {
      await service.processResults('test-analysis-id');

      expect(mockThreatRepository.create).toHaveBeenCalled();
      expect(mockThreatRepository.save).toHaveBeenCalled();
      expect(mockSandboxRepository.save).toHaveBeenCalledWith(
        expect.objectContaining({
          threat_id: expect.any(String),
        })
      );
    });

    it('should handle analysis without sandbox job ID', async () => {
      mockSandboxRepository.findOne.mockResolvedValue({
        ...mockSandboxAnalysisRecord,
        sandbox_job_id: null,
      });

      await service.processResults('test-analysis-id');

      expect(mockSandboxClient.getResults).not.toHaveBeenCalled();
    });

    it('should handle failed analysis', async () => {
      const failedResult = {
        ...mockSandboxResult,
        status: 'failed' as const,
        error: 'Analysis failed',
      };
      mockSandboxClient.getResults.mockResolvedValue(failedResult);

      await service.processResults('test-analysis-id');

      expect(mockSandboxRepository.save).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'failed',
          result_data: expect.objectContaining({
            error: 'Analysis failed',
          }),
        })
      );
    });

    it('should handle processing errors', async () => {
      mockSandboxClient.getResults.mockRejectedValue(new Error('Network error'));

      await expect(service.processResults('test-analysis-id')).rejects.toThrow();

      expect(mockSandboxRepository.save).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'failed',
        })
      );
    });

    it('should update started_at when status becomes running', async () => {
      const runningResult = {
        ...mockSandboxResult,
        status: 'running' as const,
      };
      mockSandboxClient.getResults.mockResolvedValue(runningResult);

      await service.processResults('test-analysis-id');

      expect(mockSandboxRepository.save).toHaveBeenCalledWith(
        expect.objectContaining({
          started_at: expect.any(Date),
        })
      );
    });
  });
});
