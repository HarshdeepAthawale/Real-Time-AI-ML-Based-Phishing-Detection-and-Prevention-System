import request from 'supertest';
import multer from 'multer';
import express from 'express';
import { DataSource } from 'typeorm';
import { setupSandboxRoutes } from '../../../src/routes/sandbox.routes';
import { SandboxSubmitterService } from '../../../src/services/sandbox-submitter.service';
import { ResultProcessorService } from '../../../src/services/result-processor.service';
import { SandboxQueueJob } from '../../../src/jobs/sandbox-queue.job';
import { createMockDataSource, createMockSandboxClient, createMockRedis } from '../../helpers/mocks';
import { mockFileAnalysis } from '../../fixtures/mock-data';

jest.mock('../../../src/services/file-analyzer.service');
jest.mock('../../../src/services/behavioral-analyzer.service');
jest.mock('../../../src/services/result-processor.service');
jest.mock('../../../src/jobs/sandbox-queue.job');

describe('Sandbox Routes Integration', () => {
  let app: express.Application;
  let mockDataSource: jest.Mocked<DataSource>;
  let mockSubmitterService: jest.Mocked<SandboxSubmitterService>;
  let mockResultProcessor: jest.Mocked<ResultProcessorService>;
  let mockQueueJob: jest.Mocked<SandboxQueueJob>;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    app.use(express.urlencoded({ extended: true }));

    mockDataSource = createMockDataSource();
    const mockSandboxClient = createMockSandboxClient();
    const mockRedis = createMockRedis();

    // Mock services
    const FileAnalyzerService = require('../../../src/services/file-analyzer.service').FileAnalyzerService;
    FileAnalyzerService.prototype.analyzeFile = jest.fn().mockResolvedValue(mockFileAnalysis);

    mockSubmitterService = {
      submitFile: jest.fn().mockResolvedValue('test-analysis-id'),
      submitURL: jest.fn().mockResolvedValue('test-analysis-id'),
    } as any;

    mockResultProcessor = {} as any;

    mockQueueJob = {
      addAnalysisJob: jest.fn().mockResolvedValue(undefined),
    } as any;

    // Mock repository
    const mockRepository = {
      findOne: jest.fn().mockResolvedValue({
        id: 'test-analysis-id',
        sandbox_job_id: 'test-job-id',
      }),
      findAndCount: jest.fn().mockResolvedValue([[], 0]),
    };

    mockDataSource.getRepository = jest.fn().mockReturnValue(mockRepository);

    const routes = setupSandboxRoutes(
      mockSubmitterService,
      mockResultProcessor,
      mockQueueJob,
      mockDataSource
    );

    app.use('/api/v1/sandbox', routes);
  });

  describe('POST /api/v1/sandbox/analyze/file', () => {
    it('should submit file for analysis', async () => {
      const response = await request(app)
        .post('/api/v1/sandbox/analyze/file')
        .set('x-organization-id', 'test-org-id')
        .attach('file', Buffer.from('test file'), 'test.exe');

      expect(response.status).toBe(202);
      expect(response.body).toHaveProperty('analysis_id');
      expect(response.body.status).toBe('pending');
      expect(mockSubmitterService.submitFile).toHaveBeenCalled();
    });

    it('should return 400 if no file provided', async () => {
      const response = await request(app)
        .post('/api/v1/sandbox/analyze/file')
        .set('x-organization-id', 'test-org-id');

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('No file provided');
    });
  });

  describe('POST /api/v1/sandbox/analyze/url', () => {
    it('should submit URL for analysis', async () => {
      const response = await request(app)
        .post('/api/v1/sandbox/analyze/url')
        .set('x-organization-id', 'test-org-id')
        .send({ url: 'http://example.com' });

      expect(response.status).toBe(202);
      expect(response.body).toHaveProperty('analysis_id');
      expect(response.body.status).toBe('pending');
      expect(mockSubmitterService.submitURL).toHaveBeenCalled();
    });

    it('should return 400 for invalid URL', async () => {
      const response = await request(app)
        .post('/api/v1/sandbox/analyze/url')
        .set('x-organization-id', 'test-org-id')
        .send({ url: 'not-a-valid-url' });

      expect(response.status).toBe(400);
    });

    it('should return 400 if URL is missing', async () => {
      const response = await request(app)
        .post('/api/v1/sandbox/analyze/url')
        .set('x-organization-id', 'test-org-id')
        .send({});

      expect(response.status).toBe(400);
    });
  });

  describe('GET /api/v1/sandbox/analysis/:id', () => {
    it('should get analysis results', async () => {
      const mockRepository = mockDataSource.getRepository() as any;
      mockRepository.findOne = jest.fn().mockResolvedValue({
        id: 'test-analysis-id',
        status: 'completed',
        analysis_type: 'file',
        submitted_at: new Date(),
        completed_at: new Date(),
        sandbox_provider: 'anyrun',
        sandbox_job_id: 'test-job-id',
        result_data: {
          isMalicious: true,
          threatScore: 85,
        },
        threat_id: 'test-threat-id',
        threat: {
          id: 'test-threat-id',
          threat_type: 'malware',
          severity: 'high',
          confidence_score: 85,
        },
      });

      const response = await request(app)
        .get('/api/v1/sandbox/analysis/test-analysis-id')
        .set('x-organization-id', 'test-org-id');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('analysis_id');
      expect(response.body.status).toBe('completed');
      expect(response.body.results).toBeDefined();
    });

    it('should return 404 for non-existent analysis', async () => {
      const mockRepository = mockDataSource.getRepository() as any;
      mockRepository.findOne = jest.fn().mockResolvedValue(null);

      const response = await request(app)
        .get('/api/v1/sandbox/analysis/non-existent-id')
        .set('x-organization-id', 'test-org-id');

      expect(response.status).toBe(404);
    });
  });

  describe('GET /api/v1/sandbox/analyses', () => {
    it('should list analyses with pagination', async () => {
      const mockRepository = mockDataSource.getRepository() as any;
      mockRepository.findAndCount = jest.fn().mockResolvedValue([
        [
          {
            id: 'test-analysis-id-1',
            status: 'completed',
            analysis_type: 'file',
            submitted_at: new Date(),
            completed_at: new Date(),
            threat_id: null,
          },
        ],
        1,
      ]);

      const response = await request(app)
        .get('/api/v1/sandbox/analyses?page=1&limit=20')
        .set('x-organization-id', 'test-org-id');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('analyses');
      expect(response.body).toHaveProperty('pagination');
      expect(response.body.pagination.page).toBe(1);
      expect(response.body.pagination.limit).toBe(20);
    });
  });
});
