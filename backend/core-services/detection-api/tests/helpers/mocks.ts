import { mockDeep, MockProxy } from 'jest-mock-extended';
import { Server, Socket } from 'socket.io';
import Redis from 'ioredis';
import axios, { AxiosInstance } from 'axios';

export const createMockRedis = (): MockProxy<Redis> => {
  const mockRedis = mockDeep<Redis>();
  
  // Default implementations
  mockRedis.get.mockResolvedValue(null);
  mockRedis.setex.mockResolvedValue('OK');
  mockRedis.ping.mockResolvedValue('PONG');
  mockRedis.quit.mockResolvedValue('OK');
  
  return mockRedis;
};

export const createMockAxiosInstance = (): MockProxy<AxiosInstance> => {
  return mockDeep<AxiosInstance>();
};

export const createMockSocket = (): MockProxy<Socket> => {
  const mockSocket = mockDeep<Socket>();
  mockSocket.id = 'test-socket-id';
  mockSocket.join.mockReturnValue(mockSocket as any);
  mockSocket.leave.mockReturnValue(mockSocket as any);
  mockSocket.emit.mockReturnValue(mockSocket as any);
  return mockSocket;
};

export const createMockIOServer = (): MockProxy<Server> => {
  const mockIO = mockDeep<Server>();
  mockIO.to.mockReturnValue({
    emit: jest.fn().mockReturnThis(),
  } as any);
  mockIO.emit.mockReturnValue(mockIO as any);
  return mockIO;
};

export const mockLogger = {
  info: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn(),
};
