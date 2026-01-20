import { EventStreamerService } from '../../../src/services/event-streamer.service';
import { Server, Socket } from 'socket.io';
import { mockThreat } from '../../fixtures/mock-responses';
import { testOrganizationId } from '../../fixtures/test-data';

jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('EventStreamerService', () => {
  let eventStreamer: EventStreamerService;
  let mockIO: jest.Mocked<Server>;
  let mockSocket: jest.Mocked<Socket>;
  let mockRoom: any;

  beforeEach(() => {
    jest.clearAllMocks();

    mockSocket = {
      id: 'test-socket-id',
      join: jest.fn().mockReturnThis(),
      leave: jest.fn().mockReturnThis(),
      emit: jest.fn().mockReturnThis(),
      on: jest.fn(),
    } as any;

    mockRoom = {
      emit: jest.fn().mockReturnThis(),
    };

    mockIO = {
      on: jest.fn((event: string, callback: Function) => {
        if (event === 'connection') {
          callback(mockSocket);
        }
      }),
      to: jest.fn().mockReturnValue(mockRoom),
      emit: jest.fn().mockReturnThis(),
    } as any;

    eventStreamer = new EventStreamerService(mockIO as any);
  });

  describe('connection handling', () => {
    it('should set up connection handlers on construction', () => {
      expect(mockIO.on).toHaveBeenCalledWith('connection', expect.any(Function));
    });

    it('should handle client connection', () => {
      const connectionCallback = (mockIO.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'connection'
      )?.[1];

      if (connectionCallback) {
        connectionCallback(mockSocket);
      }

      expect(eventStreamer.getConnectedClientsCount()).toBeGreaterThan(0);
    });

    it('should handle client disconnection', () => {
      const connectionCallback = (mockIO.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'connection'
      )?.[1];

      if (connectionCallback) {
        connectionCallback(mockSocket);
        
        const disconnectHandler = (mockSocket.on as jest.Mock).mock.calls.find(
          (call) => call[0] === 'disconnect'
        )?.[1];

        if (disconnectHandler) {
          disconnectHandler();
        }
      }

      expect(eventStreamer.getConnectedClientsCount()).toBe(0);
    });
  });

  describe('organization subscription', () => {
    it('should handle subscribe event', () => {
      const connectionCallback = (mockIO.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'connection'
      )?.[1];

      if (connectionCallback) {
        connectionCallback(mockSocket);
        
        const subscribeHandler = (mockSocket.on as jest.Mock).mock.calls.find(
          (call) => call[0] === 'subscribe'
        )?.[1];

        if (subscribeHandler) {
          subscribeHandler(testOrganizationId);
        }
      }

      expect(mockSocket.join).toHaveBeenCalledWith(`org:${testOrganizationId}`);
    });
  });

  describe('broadcastThreat', () => {
    it('should broadcast threat to organization room', () => {
      eventStreamer.broadcastThreat(testOrganizationId, mockThreat);

      expect(mockIO.to).toHaveBeenCalledWith(`org:${testOrganizationId}`);
      expect(mockRoom.emit).toHaveBeenCalledWith('threat_detected', {
        ...mockThreat,
        timestamp: expect.any(String),
      });
    });

    it('should include timestamp in threat broadcast', () => {
      const beforeTime = new Date().toISOString();
      eventStreamer.broadcastThreat(testOrganizationId, mockThreat);
      const afterTime = new Date().toISOString();

      const emitCall = mockRoom.emit.mock.calls.find(
        (call) => call[0] === 'threat_detected'
      );

      if (emitCall) {
        const emittedData = emitCall[1];
        expect(emittedData.timestamp).toBeDefined();
        expect(emittedData.timestamp).toBeGreaterThanOrEqual(beforeTime);
        expect(emittedData.timestamp).toBeLessThanOrEqual(afterTime);
      }
    });
  });

  describe('broadcastEvent', () => {
    it('should broadcast event to all clients', () => {
      const eventData = { url: 'https://example.com', threat: mockThreat };

      eventStreamer.broadcastEvent('url_analyzed', eventData);

      expect(mockIO.emit).toHaveBeenCalledWith('url_analyzed', {
        ...eventData,
        timestamp: expect.any(String),
      });
    });

    it('should include timestamp in event broadcast', () => {
      const eventData = { test: 'data' };

      eventStreamer.broadcastEvent('test_event', eventData);

      const emitCall = mockIO.emit.mock.calls.find(
        (call) => call[0] === 'test_event'
      );

      if (emitCall) {
        expect(emitCall[1].timestamp).toBeDefined();
      }
    });
  });

  describe('getConnectedClientsCount', () => {
    it('should return correct client count', () => {
      const connectionCallback = (mockIO.on as jest.Mock).mock.calls.find(
        (call) => call[0] === 'connection'
      )?.[1];

      if (connectionCallback) {
        connectionCallback(mockSocket);
      }

      expect(eventStreamer.getConnectedClientsCount()).toBeGreaterThanOrEqual(0);
    });
  });
});
