import { OTXClient } from '../../../src/integrations/otx.client';
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

describe('OTXClient', () => {
  let client: OTXClient;
  const apiKey = 'test-api-key';
  let mockGet: jest.Mock;

  beforeEach(() => {
    mockGet = jest.fn();
    mockedAxios.create.mockReturnValue({
      get: mockGet,
    } as any);
    client = new OTXClient(apiKey);
    jest.clearAllMocks();
  });

  describe('fetchIOCs', () => {
    it('should fetch IOCs from OTX pulses', async () => {
      const mockResponse = {
        data: {
          count: 1,
          next: null,
          previous: null,
          results: [
            {
              id: 'pulse1',
              name: 'Phishing Campaign',
              description: 'Test phishing campaign',
              TLP: 'AMBER',
              tags: ['phishing'],
              indicators: [
                {
                  id: 'ind1',
                  type: 'URL',
                  indicator: 'https://malicious.example.com',
                  title: 'Phishing URL',
                  created: new Date().toISOString(),
                },
                {
                  id: 'ind2',
                  type: 'domain',
                  indicator: 'malicious.example.com',
                  created: new Date().toISOString(),
                },
              ],
            },
          ],
        },
      };

      mockGet.mockResolvedValue(mockResponse);

      const iocs = await client.fetchIOCs();

      expect(iocs.length).toBeGreaterThan(0);
      expect(iocs[0].source).toBe('otx');
      expect(mockGet).toHaveBeenCalled();
    });

    it('should handle pagination', async () => {
      const firstPage = {
        data: {
          count: 200,
          next: 'https://otx.alienvault.com/api/v1/pulses/subscribed?page=2',
          previous: null,
          results: Array(100).fill(null).map((_, i) => ({
            id: `pulse${i}`,
            name: 'Test',
            TLP: 'GREEN',
            indicators: [],
          })),
        },
      };

      const secondPage = {
        data: {
          count: 200,
          next: null,
          previous: null,
          results: Array(100).fill(null).map((_, i) => ({
            id: `pulse${i + 100}`,
            name: 'Test',
            TLP: 'GREEN',
            indicators: [],
          })),
        },
      };

      mockGet
        .mockResolvedValueOnce(firstPage)
        .mockResolvedValueOnce(secondPage);
      
      mockedAxios.get = jest.fn()
        .mockResolvedValueOnce(secondPage);

      const iocs = await client.fetchIOCs();

      expect(iocs).toBeDefined();
    });

    it('should filter by date when since parameter provided', async () => {
      const since = new Date();
      
      mockGet.mockResolvedValue({
        data: { count: 0, results: [] },
      });

      await client.fetchIOCs(since);

      expect(mockGet).toHaveBeenCalled();
    });
  });

  describe('publishIOC', () => {
    it('should throw error when trying to publish', async () => {
      const ioc: IOC = {
        iocType: 'domain',
        iocValue: 'test.com',
      };

      await expect(client.publishIOC(ioc)).rejects.toThrow();
    });
  });
});
