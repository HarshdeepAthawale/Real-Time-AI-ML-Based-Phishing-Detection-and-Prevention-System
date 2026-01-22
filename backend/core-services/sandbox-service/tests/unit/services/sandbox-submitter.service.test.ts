import 'reflect-metadata';
import { SandboxSubmitterService } from '../../../src/services/sandbox-submitter.service';
import { FileAnalyzerService } from '../../../src/services/file-analyzer.service';
import { DataSource, Repository } from 'typeorm';
import { createMockSandboxClient } from '../../helpers/mocks';
import { mockFileAnalysis, mockFileBuffer } from '../../fixtures/mock-data';

describe('SandboxSubmitterService', () => {
  let service: SandboxSubmitterService;
  let mockSandboxClient: any;
  let mockFileAnalyzer: jest.Mocked<FileAnalyzerService>;
  let mockDataSource: jest.Mocked<DataSource>;
  let mockRepository: jest.Mocked<Repository<any>>;

  beforeEach(() => {
    mockSandboxClient = createMockSandboxClient();
    mockFileAnalyzer = {
      analyzeFile: jest.fn().mockResolvedValue(mockFileAnalysis),
    } as any;

    mockRepository = {
      create: jest.fn().mockReturnValue({
        id: 'test-analysis-id',
        organization_id: 'test-org-id',
        analysis_type: 'file',
        target_file_hash: mockFileAnalysis.hash.sha256,
        sandbox_provider: 'anyrun',
        sandbox_job_id: 'test-job-id',
        status: 'pending',
      }),
      save: jest.fn().mockResolvedValue({
        id: 'test-analysis-id',
      }),
    } as any;

    mockDataSource = {
      getRepository: jest.fn().mockReturnValue(mockRepository),
    } as any;

    service = new SandboxSubmitterService(
      mockSandboxClient,
      mockFileAnalyzer,
      mockDataSource
    );
  });

  describe('submitFile', () => {
    it('should submit file that requires sandbox analysis', async () => {
      mockSandboxClient.submitFile.mockResolvedValue('test-job-id');

      const result = await service.submitFile(
        mockFileBuffer,
        'test.exe',
        'test-org-id'
      );

      expect(result).toBe('test-analysis-id');
      expect(mockFileAnalyzer.analyzeFile).toHaveBeenCalledWith(mockFileBuffer, 'test.exe');
      expect(mockSandboxClient.submitFile).toHaveBeenCalled();
      expect(mockRepository.save).toHaveBeenCalled();
    });

    it('should create record without sandbox for files that do not require analysis', async () => {
      const nonExecutableAnalysis = {
        ...mockFileAnalysis,
        requiresSandbox: false,
      };
      mockFileAnalyzer.analyzeFile.mockResolvedValue(nonExecutableAnalysis);

      const result = await service.submitFile(
        mockFileBuffer,
        'test.txt',
        'test-org-id'
      );

      expect(result).toBe('test-analysis-id');
      expect(mockSandboxClient.submitFile).not.toHaveBeenCalled();
      expect(mockRepository.save).toHaveBeenCalled();
    });
  });

  describe('submitURL', () => {
    it('should submit URL successfully', async () => {
      mockSandboxClient.submitURL.mockResolvedValue('test-job-id');

      const result = await service.submitURL('http://example.com', 'test-org-id');

      expect(result).toBe('test-analysis-id');
      expect(mockSandboxClient.submitURL).toHaveBeenCalledWith(
        'http://example.com',
        expect.objectContaining({ timeout: expect.any(Number) })
      );
      expect(mockRepository.save).toHaveBeenCalled();
    });
  });
});
