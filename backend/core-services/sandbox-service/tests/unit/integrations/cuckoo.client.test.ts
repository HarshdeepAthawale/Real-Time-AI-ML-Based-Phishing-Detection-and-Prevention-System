import axios from 'axios';
import { CuckooClient } from '../../../src/integrations/cuckoo.client';
import FormData from 'form-data';

jest.mock('axios');
jest.mock('form-data');

const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('CuckooClient', () => {
  let client: CuckooClient;
  const baseURL = 'http://cuckoo-sandbox:8090';

  beforeEach(() => {
    jest.clearAllMocks();
    client = new CuckooClient(baseURL);
  });

  describe('submitFile', () => {
    it('should submit file successfully', async () => {
      const mockPost = jest.fn().mockResolvedValue({
        data: { task_id: 12345 },
      });

      mockedAxios.create = jest.fn().mockReturnValue({
        post: mockPost,
      } as any);

      const fileBuffer = Buffer.from('test');
      const result = await client.submitFile(fileBuffer, 'test.exe');

      expect(result).toBe('12345');
      expect(mockPost).toHaveBeenCalled();
    });

    it('should handle submission errors', async () => {
      const mockPost = jest.fn().mockRejectedValue(new Error('Submission failed'));

      mockedAxios.create = jest.fn().mockReturnValue({
        post: mockPost,
      } as any);

      await expect(
        client.submitFile(Buffer.from('test'), 'test.exe')
      ).rejects.toThrow();
    });
  });

  describe('submitURL', () => {
    it('should submit URL successfully', async () => {
      const mockPost = jest.fn().mockResolvedValue({
        data: { task_id: 67890 },
      });

      mockedAxios.create = jest.fn().mockReturnValue({
        post: mockPost,
      } as any);

      const result = await client.submitURL('http://example.com');

      expect(result).toBe('67890');
      expect(mockPost).toHaveBeenCalledWith('/tasks/create/url', { url: 'http://example.com' });
    });
  });

  describe('getStatus', () => {
    it('should get status successfully', async () => {
      const mockGet = jest.fn().mockResolvedValue({
        data: {
          task: {
            id: 12345,
            status: 'running',
          },
        },
      });

      mockedAxios.create = jest.fn().mockReturnValue({
        get: mockGet,
      } as any);

      const result = await client.getStatus('12345');

      expect(result.jobId).toBe('12345');
      expect(result.status).toBe('running');
    });

    it('should map status correctly', async () => {
      const statuses = ['pending', 'running', 'completed', 'reported', 'failed'];
      
      for (const status of statuses) {
        const mockGet = jest.fn().mockResolvedValue({
          data: {
            task: {
              id: 12345,
              status,
            },
          },
        });

        mockedAxios.create = jest.fn().mockReturnValue({
          get: mockGet,
        } as any);

        const result = await client.getStatus('12345');
        expect(result.status).toBeDefined();
      }
    });
  });

  describe('getResults', () => {
    it('should extract results correctly', async () => {
      const mockGet = jest.fn().mockResolvedValue({
        data: {
          network: [
            {
              protocol: 'tcp',
              dst: '192.168.1.1',
              dport: 4444,
            },
          ],
          target: {
            file: {
              path: '/path/to/file',
              type: 'exe',
            },
          },
          behavior: {
            processes: [
              {
                process_name: 'test.exe',
                pid: 1234,
                command_line: 'test.exe',
                parent_id: 1,
              },
            ],
          },
          screenshots: [{ path: 'screenshot.png' }],
          signatures: [{ name: 'malware' }],
          info: {
            score: 8.5,
          },
        },
      });

      mockedAxios.create = jest.fn().mockReturnValue({
        get: mockGet,
      } as any);

      const result = await client.getResults('12345');

      expect(result.status).toBe('completed');
      expect(result.results?.network).toBeDefined();
      expect(result.results?.processes).toBeDefined();
      expect(result.results?.score).toBe(8.5);
    });
  });
});
