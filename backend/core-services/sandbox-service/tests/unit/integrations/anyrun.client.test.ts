import axios from 'axios';
import { AnyRunClient } from '../../../src/integrations/anyrun.client';
import FormData from 'form-data';

jest.mock('axios');
jest.mock('form-data');

const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('AnyRunClient', () => {
  let client: AnyRunClient;
  const apiKey = 'test-api-key';

  beforeEach(() => {
    jest.clearAllMocks();
    client = new AnyRunClient(apiKey);
  });

  describe('submitFile', () => {
    it('should submit file successfully', async () => {
      const mockPost = jest.fn().mockResolvedValue({
        data: {
          data: {
            taskid: 'anyrun-task-123',
          },
        },
      });

      mockedAxios.create = jest.fn().mockReturnValue({
        post: mockPost,
      } as any);

      const fileBuffer = Buffer.from('test');
      const result = await client.submitFile(fileBuffer, 'test.exe');

      expect(result).toBe('anyrun-task-123');
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
        data: {
          data: {
            taskid: 'anyrun-url-123',
          },
        },
      });

      mockedAxios.create = jest.fn().mockReturnValue({
        post: mockPost,
      } as any);

      const result = await client.submitURL('http://example.com');

      expect(result).toBe('anyrun-url-123');
      expect(mockPost).toHaveBeenCalled();
    });
  });

  describe('getStatus', () => {
    it('should get status successfully', async () => {
      const mockGet = jest.fn().mockResolvedValue({
        data: {
          data: {
            status: 'done',
          },
        },
      });

      mockedAxios.create = jest.fn().mockReturnValue({
        get: mockGet,
      } as any);

      const result = await client.getStatus('task-123');

      expect(result.jobId).toBe('task-123');
      expect(result.status).toBe('completed');
    });

    it('should map status correctly', async () => {
      const statusMap = [
        { input: 'done', expected: 'completed' },
        { input: 'running', expected: 'running' },
        { input: 'processing', expected: 'running' },
        { input: 'failed', expected: 'failed' },
        { input: 'error', expected: 'failed' },
        { input: 'pending', expected: 'pending' },
      ];

      for (const { input, expected } of statusMap) {
        const mockGet = jest.fn().mockResolvedValue({
          data: {
            data: {
              status: input,
            },
          },
        });

        mockedAxios.create = jest.fn().mockReturnValue({
          get: mockGet,
        } as any);

        const result = await client.getStatus('task-123');
        expect(result.status).toBe(expected);
      }
    });
  });

  describe('getResults', () => {
    it('should extract results correctly', async () => {
      const mockGet = jest.fn().mockResolvedValue({
        data: {
          data: {
            status: 'done',
            network: {
              connections: [
                {
                  protocol: 'tcp',
                  ip: '192.168.1.1',
                  port: 4444,
                },
              ],
              http: [
                {
                  host: 'example.com',
                  port: 80,
                  method: 'GET',
                  path: '/',
                  status: 200,
                },
              ],
            },
            files: [
              {
                path: '/path/to/file',
                action: 'created',
                type: 'exe',
              },
            ],
            processes: [
              {
                name: 'test.exe',
                pid: 1234,
                cmdline: 'test.exe',
                ppid: 1,
              },
            ],
            screenshots: [{ url: 'screenshot.png' }],
            threats: [
              { name: 'malware' },
            ],
            verdict: 'malicious',
          },
        },
      });

      mockedAxios.create = jest.fn().mockReturnValue({
        get: mockGet,
      } as any);

      const result = await client.getResults('task-123');

      expect(result.status).toBe('completed');
      expect(result.results?.network).toBeDefined();
      expect(result.results?.filesystem).toBeDefined();
      expect(result.results?.processes).toBeDefined();
      expect(result.results?.signatures).toBeDefined();
    });

    it('should handle incomplete analysis', async () => {
      const mockGet = jest.fn().mockResolvedValue({
        data: {
          data: {
            status: 'running',
          },
        },
      });

      mockedAxios.create = jest.fn().mockReturnValue({
        get: mockGet,
      } as any);

      const result = await client.getResults('task-123');

      expect(result.status).toBe('running');
      expect(result.results).toBeUndefined();
    });
  });
});
