import { Server, Socket } from 'socket.io';
import { EventStreamerService } from '../services/event-streamer.service';
import { logger } from '../utils/logger';

interface AuthenticatedSocket extends Socket {
  apiKey?: string;
  organizationId?: string;
}

export function setupWebSocket(
  io: Server,
  eventStreamer: EventStreamerService
): void {
  // Middleware for authentication
  io.use((socket: AuthenticatedSocket, next) => {
    const apiKey = socket.handshake.auth?.apiKey || socket.handshake.headers['x-api-key'] as string;
    const organizationId = socket.handshake.auth?.organizationId || socket.handshake.headers['x-organization-id'] as string;

    // For development, allow connections without API key
    // In production, this should validate against database
    if (!apiKey && process.env.NODE_ENV === 'production') {
      logger.warn(`WebSocket connection rejected: no API key`, { socketId: socket.id });
      return next(new Error('Authentication required'));
    }

    // Attach auth info to socket
    socket.apiKey = apiKey;
    socket.organizationId = organizationId;

    logger.debug(`WebSocket authentication`, {
      socketId: socket.id,
      hasApiKey: !!apiKey,
      organizationId: organizationId || 'none'
    });

    next();
  });

  io.on('connection', (socket: AuthenticatedSocket) => {
    logger.info(`WebSocket client connected: ${socket.id}`, {
      organizationId: socket.organizationId || 'none'
    });
    
    socket.on('subscribe', (organizationId: string) => {
      // Use provided organizationId or fall back to authenticated one
      const orgId = organizationId || socket.organizationId;
      
      if (!orgId) {
        socket.emit('error', { message: 'Organization ID is required' });
        logger.warn(`Subscribe failed: no organization ID`, { socketId: socket.id });
        return;
      }
      
      socket.join(`org:${orgId}`);
      logger.info(`Client ${socket.id} subscribed to org ${orgId}`);
      socket.emit('subscribed', { organizationId: orgId });
    });
    
    socket.on('unsubscribe', (organizationId: string) => {
      const orgId = organizationId || socket.organizationId;
      if (orgId) {
        socket.leave(`org:${orgId}`);
        logger.info(`Client ${socket.id} unsubscribed from org ${orgId}`);
        socket.emit('unsubscribed', { organizationId: orgId });
      }
    });
    
    socket.on('ping', () => {
      socket.emit('pong', { timestamp: new Date().toISOString() });
    });
    
    socket.on('disconnect', (reason) => {
      logger.info(`WebSocket client disconnected: ${socket.id}`, { reason });
    });

    // Handle authentication errors
    socket.on('error', (error) => {
      logger.error(`WebSocket error for ${socket.id}`, { error });
    });
  });
}
