import { BehavioralAnalyzerService } from '../../../src/services/behavioral-analyzer.service';
import { SandboxResult } from '../../../src/integrations/base-sandbox.client';
import {
  mockSandboxResult,
  mockNetworkActivity,
  mockFileSystemActivity,
  mockProcessActivity,
} from '../../fixtures/mock-data';

describe('BehavioralAnalyzerService', () => {
  let service: BehavioralAnalyzerService;

  beforeEach(() => {
    service = new BehavioralAnalyzerService();
  });

  describe('analyze', () => {
    it('should identify malicious behavior with high threat score', () => {
      const result = service.analyze(mockSandboxResult);

      expect(result.isMalicious).toBe(true);
      expect(result.threatScore).toBeGreaterThanOrEqual(50);
      expect(result.indicators.length).toBeGreaterThan(0);
    });

    it('should detect C2 communication indicators', () => {
      const sandboxResult: SandboxResult = {
        jobId: 'test',
        status: 'completed',
        results: {
          network: mockNetworkActivity,
        },
      };

      const result = service.analyze(sandboxResult);

      expect(result.networkActivity.suspiciousConnections).toBeGreaterThan(0);
    });

    it('should detect data exfiltration', () => {
      const sandboxResult: SandboxResult = {
        jobId: 'test',
        status: 'completed',
        results: {
          network: [
            {
              protocol: 'http',
              destination: 'example.com',
              port: 80,
              method: 'POST',
              statusCode: 200,
            },
          ],
        },
      };

      const result = service.analyze(sandboxResult);

      expect(result.networkActivity.dataExfiltration).toBe(true);
      expect(result.indicators).toContain('data_exfiltration');
    });

    it('should detect system file access', () => {
      const sandboxResult: SandboxResult = {
        jobId: 'test',
        status: 'completed',
        results: {
          filesystem: mockFileSystemActivity,
        },
      };

      const result = service.analyze(sandboxResult);

      expect(result.fileSystemActivity.systemFileAccess).toBe(true);
      expect(result.indicators).toContain('system_file_access');
    });

    it('should detect suspicious processes', () => {
      const sandboxResult: SandboxResult = {
        jobId: 'test',
        status: 'completed',
        results: {
          processes: mockProcessActivity,
        },
      };

      const result = service.analyze(sandboxResult);

      expect(result.processActivity.suspiciousProcesses).toBeGreaterThan(0);
      expect(result.indicators).toContain('suspicious_processes');
    });

    it('should detect malware signatures', () => {
      const sandboxResult: SandboxResult = {
        jobId: 'test',
        status: 'completed',
        results: {
          signatures: ['malware.signature', 'trojan.signature'],
        },
      };

      const result = service.analyze(sandboxResult);

      expect(result.indicators).toContain('malware_signatures_detected');
      expect(result.threatScore).toBeGreaterThanOrEqual(25);
    });

    it('should detect high sandbox score', () => {
      const sandboxResult: SandboxResult = {
        jobId: 'test',
        status: 'completed',
        results: {
          score: 8.5,
        },
      };

      const result = service.analyze(sandboxResult);

      expect(result.indicators).toContain('high_sandbox_score');
      expect(result.threatScore).toBeGreaterThanOrEqual(30);
    });

    it('should return non-malicious for clean results', () => {
      const sandboxResult: SandboxResult = {
        jobId: 'test',
        status: 'completed',
        results: {
          network: [],
          filesystem: [],
          processes: [],
          score: 1,
        },
      };

      const result = service.analyze(sandboxResult);

      expect(result.isMalicious).toBe(false);
      expect(result.threatScore).toBeLessThan(50);
    });

    it('should cap threat score at 100', () => {
      const sandboxResult: SandboxResult = {
        jobId: 'test',
        status: 'completed',
        results: {
          network: Array(10).fill(mockNetworkActivity[0]),
          filesystem: Array(10).fill(mockFileSystemActivity[0]),
          processes: Array(10).fill(mockProcessActivity[0]),
          signatures: ['sig1', 'sig2', 'sig3'],
          score: 10,
        },
      };

      const result = service.analyze(sandboxResult);

      expect(result.threatScore).toBeLessThanOrEqual(100);
    });
  });
});
