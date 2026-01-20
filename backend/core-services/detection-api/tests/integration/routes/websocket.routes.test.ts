import { createServer } from 'http';
import { Server } from 'socket.io';
import { io as Client, Socket as ClientSocket } from 'socket.io-client';
import express from 'express';
import { setupWebSocket } from '../../../src/routes/websocket.routes';
import { EventStreamerService } from '../../../src/services/event-streamer.service';
import { testOrganizationId } from '../../fixtures/test-data';
import { mockThreat } from '../../fixtures/mock-responses';

jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('WebSocket Routes Integration', () => {
  let httpServer: any;
  let io: Server;
  let eventStreamer: EventStreamerService;
  let clientSocket: ClientSocket;
  let port: number;

  beforeAll((done) => {
    const app = express();
    httpServer = createServer(app);
    io = new Server(httpServer);
    eventStreamer = new EventStreamerService(io);
    setupWebSocket(io, eventStreamer);

    httpServer.listen(() => {
      port = (httpServer.address() as any).port;
      done();
    });
  });

  afterAll(() => {
    if (clientSocket) {
      clientSocket.close();
    }
    io.close();
    httpServer.close();
  });

  beforeEach((done) => {
    clientSocket = Client(`http://localhost:${port}`, {
      transports: ['websocket'],
    });

    clientSocket.on('connect', () => {
      done();
    });
  });

  afterEach((done) => {
    if (clientSocket.connected) {
      clientSocket.disconnect();
    }
    done();
  });

  describe('connection', () => {
    it('should establish WebSocket connection', (done) => {
      expect(clientSocket.connected).toBe(true);
      done();
    });
  });

  describe('subscribe', () => {
    it('should subscribe to organization room', (done) => {
      clientSocket.emit('subscribe', testOrganizationId);

      clientSocket.on('subscribed', (data) => {
        expect(data.organizationId).toBe(testOrganizationId);
        done();
      });
    });

    it('should handle missing organization ID', (done) => {
      clientSocket.emit('subscribe', '');

      clientSocket.on('error', (error) => {
        expect(error.message).toContain('Organization ID is required');
        done();
      });
    });
  });

  describe('unsubscribe', () => {
    it('should unsubscribe from organization room', (done) => {
      clientSocket.emit('subscribe', testOrganizationId);
      
      setTimeout(() => {
        clientSocket.emit('unsubscribe', testOrganizationId);

        clientSocket.on('unsubscribed', (data) => {
          expect(data.organizationId).toBe(testOrganizationId);
          done();
        });
      }, 100);
    });
  });

  describe('ping/pong', () => {
    it('should respond to ping with pong', (done) => {
      clientSocket.emit('ping');

      clientSocket.on('pong', (data) => {
        expect(data.timestamp).toBeDefined();
        done();
      });
    });
  });

  describe('threat detection events', () => {
    it('should receive threat_detected event for subscribed organization', (done) => {
      clientSocket.emit('subscribe', testOrganizationId);

      setTimeout(() => {
        eventStreamer.broadcastThreat(testOrganizationId, mockThreat);

        clientSocket.on('threat_detected', (data) => {
          expect(data.isThreat).toBe(mockThreat.isThreat);
          expect(data.severity).toBe(mockThreat.severity);
          done();
        });
      }, 100);
    });

    it('should not receive threat_detected for unsubscribed organization', (done) => {
      const otherOrgId = '999e9999-e99b-99d9-a999-999999999999';
      eventStreamer.broadcastThreat(otherOrgId, mockThreat);

      const timeout = setTimeout(() => {
        // If we reach here, no event was received (expected)
        done();
      }, 500);

      clientSocket.on('threat_detected', () => {
        clearTimeout(timeout);
        done.fail('Should not receive event for unsubscribed org');
      });
    });
  });

  describe('url_analyzed events', () => {
    it('should receive url_analyzed event', (done) => {
      eventStreamer.broadcastEvent('url_analyzed', {
        url: 'https://example.com',
        threat: mockThreat,
      });

      clientSocket.on('url_analyzed', (data) => {
        expect(data.url).toBe('https://example.com');
        expect(data.threat).toBeDefined();
        done();
      });
    });
  });
});
