import request from 'supertest';
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import detectionRoutes from '../../../src/routes/detection.routes';
import { EventStreamerService } from '../../../src/services/event-streamer.service';
import { setEventStreamer } from '../../../src/routes/detection.routes';
import {
  sampleEmailContent,
  suspiciousURL,
  sampleText,
  testApiKey,
  testOrganizationId,
} from '../../fixtures/test-data';
import { mockThreat } from '../../fixtures/mock-responses';

// Mock services
jest.mock('../../../src/services/orchestrator.service');
jest.mock('../../../src/services/decision-engine.service');
jest.mock('../../../src/services/cache.service');
jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

import { OrchestratorService } from '../../../src/services/orchestrator.service';
import { DecisionEngineService } from '../../../src/services/decision-engine.service';
import { CacheService } from '../../../src/services/cache.service';

describe('Detection Routes Integration', () => {
  let app: express.Application;
  let httpServer: any;
  let io: Server;
  let eventStreamer: EventStreamerService;
  let mockOrchestrator: jest.Mocked<OrchestratorService>;
  let mockDecisionEngine: jest.Mocked<DecisionEngineService>;
  let mockCacheService: jest.Mocked<CacheService>;

  beforeEach(() => {
    jest.clearAllMocks();

    app = express();
    app.use(express.json());
    httpServer = createServer(app);
    io = new Server(httpServer);
    eventStreamer = new EventStreamerService(io);
    setEventStreamer(eventStreamer);

    app.use('/api/v1/detect', detectionRoutes);

    // Setup mocks
    mockOrchestrator = {
      analyzeEmail: jest.fn(),
      analyzeURL: jest.fn(),
      analyzeText: jest.fn(),
    } as any;

    mockDecisionEngine = {
      makeDecision: jest.fn().mockReturnValue(mockThreat),
    } as any;

    mockCacheService = {
      get: jest.fn(),
      set: jest.fn(),
      generateCacheKey: jest.fn().mockReturnValue('test-cache-key'),
    } as any;

    // Replace service instances
    (OrchestratorService as jest.MockedClass<typeof OrchestratorService>).mockImplementation(
      () => mockOrchestrator
    );
    (DecisionEngineService as jest.MockedClass<typeof DecisionEngineService>).mockImplementation(
      () => mockDecisionEngine
    );
    (CacheService as jest.MockedClass<typeof CacheService>).mockImplementation(
      () => mockCacheService
    );
  });

  afterEach(() => {
    httpServer.close();
  });

  describe('POST /api/v1/detect/email', () => {
    it('should detect email threat successfully', async () => {
      mockCacheService.get.mockResolvedValue(null);
      mockOrchestrator.analyzeEmail.mockResolvedValue({
        nlp: { phishing_probability: 0.8 },
        url: null,
        visual: null,
        processingTimeMs: 100,
      });

      const response = await request(app)
        .post('/api/v1/detect/email')
        .set('x-api-key', testApiKey)
        .send({
          emailContent: sampleEmailContent,
        });

      expect(response.status).toBe(200);
      expect(response.body.isThreat).toBeDefined();
      expect(mockOrchestrator.analyzeEmail).toHaveBeenCalled();
      expect(mockDecisionEngine.makeDecision).toHaveBeenCalled();
    });

    it('should return cached result when available', async () => {
      mockCacheService.get.mockResolvedValue(mockThreat);

      const response = await request(app)
        .post('/api/v1/detect/email')
        .set('x-api-key', testApiKey)
        .send({
          emailContent: sampleEmailContent,
        });

      expect(response.status).toBe(200);
      expect(response.body.cached).toBe(true);
      expect(mockOrchestrator.analyzeEmail).not.toHaveBeenCalled();
    });

    it('should require authentication', async () => {
      const response = await request(app)
        .post('/api/v1/detect/email')
        .send({
          emailContent: sampleEmailContent,
        });

      expect(response.status).toBe(401);
    });

    it('should validate request body', async () => {
      const response = await request(app)
        .post('/api/v1/detect/email')
        .set('x-api-key', testApiKey)
        .send({
          emailContent: '', // Invalid: empty string
        });

      expect(response.status).toBe(400);
      expect(response.body.error).toBe('Validation failed');
    });

    it('should broadcast threat event when organization ID provided', async () => {
      const broadcastSpy = jest.spyOn(eventStreamer, 'broadcastThreat');
      mockCacheService.get.mockResolvedValue(null);
      mockOrchestrator.analyzeEmail.mockResolvedValue({
        nlp: { phishing_probability: 0.8 },
        url: null,
        visual: null,
        processingTimeMs: 100,
      });

      await request(app)
        .post('/api/v1/detect/email')
        .set('x-api-key', testApiKey)
        .set('x-organization-id', testOrganizationId)
        .send({
          emailContent: sampleEmailContent,
          organizationId: testOrganizationId,
        });

      expect(broadcastSpy).toHaveBeenCalledWith(
        testOrganizationId,
        expect.any(Object)
      );
    });
  });

  describe('POST /api/v1/detect/url', () => {
    it('should detect URL threat successfully', async () => {
      mockCacheService.get.mockResolvedValue(null);
      mockOrchestrator.analyzeURL.mockResolvedValue({
        nlp: null,
        url: { phishing_probability: 0.9 },
        visual: null,
        processingTimeMs: 150,
      });

      const response = await request(app)
        .post('/api/v1/detect/url')
        .set('x-api-key', testApiKey)
        .send({
          url: suspiciousURL,
        });

      expect(response.status).toBe(200);
      expect(response.body.isThreat).toBeDefined();
      expect(mockOrchestrator.analyzeURL).toHaveBeenCalled();
    });

    it('should handle legitimate URL comparison', async () => {
      mockCacheService.get.mockResolvedValue(null);
      mockOrchestrator.analyzeURL.mockResolvedValue({
        nlp: null,
        url: { phishing_probability: 0.1 },
        visual: null,
        processingTimeMs: 150,
      });

      const response = await request(app)
        .post('/api/v1/detect/url')
        .set('x-api-key', testApiKey)
        .send({
          url: suspiciousURL,
          legitimateUrl: 'https://legitimate.com',
        });

      expect(response.status).toBe(200);
      expect(mockOrchestrator.analyzeURL).toHaveBeenCalledWith(
        expect.objectContaining({
          legitimateUrl: 'https://legitimate.com',
        })
      );
    });

    it('should validate URL format', async () => {
      const response = await request(app)
        .post('/api/v1/detect/url')
        .set('x-api-key', testApiKey)
        .send({
          url: 'not-a-valid-url',
        });

      expect(response.status).toBe(400);
    });
  });

  describe('POST /api/v1/detect/text', () => {
    it('should detect text threat successfully', async () => {
      mockCacheService.get.mockResolvedValue(null);
      mockOrchestrator.analyzeText.mockResolvedValue({
        nlp: { phishing_probability: 0.75 },
        url: null,
        visual: null,
        processingTimeMs: 80,
      });

      const response = await request(app)
        .post('/api/v1/detect/text')
        .set('x-api-key', testApiKey)
        .send({
          text: sampleText,
        });

      expect(response.status).toBe(200);
      expect(response.body.isThreat).toBeDefined();
      expect(mockOrchestrator.analyzeText).toHaveBeenCalled();
    });

    it('should validate text content', async () => {
      const response = await request(app)
        .post('/api/v1/detect/text')
        .set('x-api-key', testApiKey)
        .send({
          text: '', // Invalid: empty string
        });

      expect(response.status).toBe(400);
    });
  });
});
