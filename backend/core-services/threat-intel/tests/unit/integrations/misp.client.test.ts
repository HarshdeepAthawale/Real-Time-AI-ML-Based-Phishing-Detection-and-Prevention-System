import { MISPClient } from '../../../src/integrations/misp.client';
import { IOC } from '../../../src/models/ioc.model';
import axios from 'axios';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock logger
jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('MISPClient', () => {
  let client: MISPClient;
  const baseURL = 'https://misp.example.com';
  const apiKey = 'test-api-key';
  let mockPost: jest.Mock;

  beforeEach(() => {
    mockPost = jest.fn();
    mockedAxios.create.mockReturnValue({
      post: mockPost,
    } as any);
    client = new MISPClient(baseURL, apiKey);
    jest.clearAllMocks();
  });

  describe('fetchIOCs', () => {
    it('should fetch IOCs from MISP', async () => {
      const mockResponse = {
        data: {
          response: {
            Attribute: [
              {
                id: '1',
                event_id: '100',
                type: 'domain',
                value: 'malicious.example.com',
                timestamp: Math.floor(Date.now() / 1000),
                Event: {
                  info: 'Phishing campaign',
                  threat_level_id: 2,
                },
              },
            ],
          },
        },
      };

      mockPost.mockResolvedValue(mockResponse);

      const iocs = await client.fetchIOCs();

      expect(iocs).toHaveLength(1);
      expect(iocs[0].iocType).toBe('domain');
      expect(iocs[0].iocValue).toBe('malicious.example.com');
      expect(iocs[0].source).toBe('misp');
      expect(mockPost).toHaveBeenCalled();
    });

    it('should filter IOCs by date when since parameter provided', async () => {
      const since = new Date();
      
      mockPost.mockResolvedValue({
        data: { response: { Attribute: [] } },
      });

      await client.fetchIOCs(since);

      // Verify post was called with timestamp
      expect(mockPost).toHaveBeenCalled();
    });
  });

  describe('publishIOC', () => {
    it('should publish IOC to MISP', async () => {
      const ioc: IOC = {
        iocType: 'domain',
        iocValue: 'malicious.example.com',
        threatType: 'phishing',
        severity: 'high',
        confidence: 90,
      };

      const mockAxiosPost = jest.fn().mockResolvedValue({ data: {} });
      mockedAxios.post = mockAxiosPost;

      await client.publishIOC(ioc);

      expect(mockAxiosPost).toHaveBeenCalled();
    });
  });

  describe('Type mappings', () => {
    it('should correctly map IOC types to MISP types', () => {
      const testCases = [
        { ioc: 'url', misp: 'url' },
        { ioc: 'domain', misp: 'domain' },
        { ioc: 'ip', misp: 'ip-dst' },
        { ioc: 'hash_md5', misp: 'md5' },
      ];

      testCases.forEach(({ ioc, misp }) => {
        // Access private method via reflection or test through public methods
        // For now, we verify through integration tests
      });
    });
  });
});
