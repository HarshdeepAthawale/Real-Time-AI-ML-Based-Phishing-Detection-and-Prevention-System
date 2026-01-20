import { OrchestratorService } from '../../../src/services/orchestrator.service';
import axios from 'axios';
import { mockNLPResponse, mockURLResponse } from '../../fixtures/mock-responses';
import { sampleEmailContent, suspiciousURL } from '../../fixtures/test-data';

jest.mock('axios');
jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('OrchestratorService', () => {
  let orchestrator: OrchestratorService;
  let mockNlpClient: any;
  let mockUrlClient: any;
  let mockVisualClient: any;

  beforeEach(() => {
    jest.clearAllMocks();
    
    mockNlpClient = {
      post: jest.fn(),
    };
    
    mockUrlClient = {
      post: jest.fn(),
    };
    
    mockVisualClient = {
      post: jest.fn(),
    };

    mockedAxios.create = jest.fn((config: any) => {
      if (config.baseURL?.includes('nlp')) {
        return mockNlpClient;
      } else if (config.baseURL?.includes('url')) {
        return mockUrlClient;
      } else if (config.baseURL?.includes('visual')) {
        return mockVisualClient;
      }
      return {} as any;
    });

    orchestrator = new OrchestratorService();
  });

  describe('analyzeEmail', () => {
    it('should analyze email with NLP and extract URLs', async () => {
      mockNlpClient.post.mockResolvedValue({ data: mockNLPResponse });
      mockUrlClient.post.mockResolvedValue({ data: mockURLResponse });

      const result = await orchestrator.analyzeEmail({
        emailContent: sampleEmailContent,
        includeFeatures: false,
      });

      expect(result.nlp).toEqual(mockNLPResponse);
      expect(result.url).toBeTruthy();
      expect(result.visual).toBeNull();
      expect(result.processingTimeMs).toBeGreaterThan(0);
      expect(mockNlpClient.post).toHaveBeenCalledWith('/api/v1/analyze-email', {
        raw_email: sampleEmailContent,
        include_features: false,
      });
    });

    it('should handle NLP service errors gracefully', async () => {
      mockNlpClient.post.mockRejectedValue(new Error('NLP service unavailable'));
      mockUrlClient.post.mockResolvedValue({ data: mockURLResponse });

      const result = await orchestrator.analyzeEmail({
        emailContent: sampleEmailContent,
      });

      expect(result.nlp).toBeNull();
      expect(result.url).toBeTruthy();
    });

    it('should limit URL extraction to 5 URLs', async () => {
      const emailWithManyUrls = sampleEmailContent + ' ' + 
        Array(10).fill('https://example.com').join(' ');
      
      mockNlpClient.post.mockResolvedValue({ data: mockNLPResponse });
      mockUrlClient.post.mockResolvedValue({ data: mockURLResponse });

      await orchestrator.analyzeEmail({
        emailContent: emailWithManyUrls,
      });

      expect(mockUrlClient.post).toHaveBeenCalledTimes(5);
    });

    it('should handle email with no URLs', async () => {
      const emailNoUrls = 'This is a simple email with no URLs.';
      mockNlpClient.post.mockResolvedValue({ data: mockNLPResponse });

      const result = await orchestrator.analyzeEmail({
        emailContent: emailNoUrls,
      });

      expect(result.nlp).toEqual(mockNLPResponse);
      expect(result.url).toBeNull();
    });
  });

  describe('analyzeURL', () => {
    it('should analyze URL with all services', async () => {
      const mockVisualResponse = {
        dom_analysis: { text: 'Sample page text' },
      };
      
      mockUrlClient.post.mockResolvedValue({ data: mockURLResponse });
      mockVisualClient.post.mockResolvedValue({ data: mockVisualResponse });
      mockNlpClient.post.mockResolvedValue({ data: mockNLPResponse });

      const result = await orchestrator.analyzeURL({
        url: suspiciousURL,
        legitimateDomain: 'example.com',
      });

      expect(result.url).toEqual(mockURLResponse);
      expect(result.visual).toEqual(mockVisualResponse);
      expect(result.nlp).toEqual(mockNLPResponse);
      expect(mockUrlClient.post).toHaveBeenCalledWith('/api/v1/analyze-url', {
        url: suspiciousURL,
        legitimate_domain: 'example.com',
      });
    });

    it('should handle visual service failure gracefully', async () => {
      mockUrlClient.post.mockResolvedValue({ data: mockURLResponse });
      mockVisualClient.post.mockRejectedValue(new Error('Visual service error'));

      const result = await orchestrator.analyzeURL({
        url: suspiciousURL,
      });

      expect(result.url).toEqual(mockURLResponse);
      expect(result.visual).toBeNull();
    });

    it('should skip NLP analysis if no page text available', async () => {
      const mockVisualResponse = {
        dom_analysis: {},
      };
      
      mockUrlClient.post.mockResolvedValue({ data: mockURLResponse });
      mockVisualClient.post.mockResolvedValue({ data: mockVisualResponse });

      const result = await orchestrator.analyzeURL({
        url: suspiciousURL,
      });

      expect(result.nlp).toBeNull();
      expect(mockNlpClient.post).not.toHaveBeenCalled();
    });
  });

  describe('analyzeText', () => {
    it('should analyze text with NLP service', async () => {
      mockNlpClient.post.mockResolvedValue({ data: mockNLPResponse });

      const result = await orchestrator.analyzeText({
        text: 'Sample text to analyze',
        includeFeatures: true,
      });

      expect(result.nlp).toEqual(mockNLPResponse);
      expect(result.url).toBeNull();
      expect(result.visual).toBeNull();
      expect(mockNlpClient.post).toHaveBeenCalledWith('/api/v1/analyze-text', {
        text: 'Sample text to analyze',
        include_features: true,
      });
    });

    it('should handle NLP service errors', async () => {
      mockNlpClient.post.mockRejectedValue(new Error('NLP service error'));

      const result = await orchestrator.analyzeText({
        text: 'Sample text',
      });

      expect(result.nlp).toBeNull();
    });
  });
});
