import { Server, Socket } from 'socket.io';
import { EventStreamerService } from '../services/event-streamer.service';
import { logger } from '../utils/logger';

export function setupWebSocket(
  io: Server,
  eventStreamer: EventStreamerService
): void {
  io.on('connection', (socket: Socket) => {
    logger.info(`WebSocket client connected: ${socket.id}`);
    
    socket.on('subscribe', (organizationId: string) => {
      if (!organizationId) {
        socket.emit('error', { message: 'Organization ID is required' });
        return;
      }
      socket.join(`org:${organizationId}`);
      logger.info(`Client ${socket.id} subscribed to org ${organizationId}`);
      socket.emit('subscribed', { organizationId });
    });
    
    socket.on('unsubscribe', (organizationId: string) => {
      socket.leave(`org:${organizationId}`);
      logger.info(`Client ${socket.id} unsubscribed from org ${organizationId}`);
      socket.emit('unsubscribed', { organizationId });
    });
    
    socket.on('ping', () => {
      socket.emit('pong', { timestamp: new Date().toISOString() });
    });
    
    socket.on('disconnect', () => {
      logger.info(`WebSocket client disconnected: ${socket.id}`);
    });
  });
}
