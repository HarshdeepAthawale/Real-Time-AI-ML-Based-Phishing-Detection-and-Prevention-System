import { Server, Socket } from 'socket.io';
import { logger } from '../utils/logger';
import { Threat } from '../models/detection.model';

export class EventStreamerService {
  private io: Server;
  private connectedClients: Map<string, Socket> = new Map();
  
  constructor(io: Server) {
    this.io = io;
    this.setupConnectionHandlers();
  }
  
  private setupConnectionHandlers(): void {
    this.io.on('connection', (socket: Socket) => {
      logger.info(`Client connected: ${socket.id}`);
      this.connectedClients.set(socket.id, socket);
      
      socket.on('disconnect', () => {
        logger.info(`Client disconnected: ${socket.id}`);
        this.connectedClients.delete(socket.id);
      });
      
      socket.on('subscribe', (organizationId: string) => {
        socket.join(`org:${organizationId}`);
        logger.info(`Client ${socket.id} subscribed to org ${organizationId}`);
      });
      
      socket.on('unsubscribe', (organizationId: string) => {
        socket.leave(`org:${organizationId}`);
        logger.info(`Client ${socket.id} unsubscribed from org ${organizationId}`);
      });
    });
  }
  
  broadcastThreat(organizationId: string, threat: Threat): void {
    if (!organizationId) {
      logger.debug('Skipping threat broadcast: no organization ID');
      return;
    }
    
    try {
      const room = `org:${organizationId}`;
      const eventData = {
        ...threat,
        timestamp: new Date().toISOString()
      };
      
      this.io.to(room).emit('threat_detected', eventData);
      
      const roomSize = this.io.sockets.adapter.rooms.get(room)?.size || 0;
      logger.debug(`Broadcasted threat to org ${organizationId} (${roomSize} clients)`);
    } catch (error: any) {
      logger.error('Failed to broadcast threat event:', error.message);
    }
  }
  
  broadcastEvent(eventType: string, data: any): void {
    try {
      const eventData = {
        ...data,
        timestamp: new Date().toISOString()
      };
      
      this.io.emit(eventType, eventData);
      logger.debug(`Broadcasted event: ${eventType} to all clients`);
    } catch (error: any) {
      logger.error(`Failed to broadcast event ${eventType}:`, error.message);
    }
  }
  
  broadcastToOrganization(organizationId: string, eventType: string, data: any): void {
    if (!organizationId) {
      logger.debug(`Skipping ${eventType} broadcast: no organization ID`);
      return;
    }
    
    try {
      const room = `org:${organizationId}`;
      const eventData = {
        ...data,
        timestamp: new Date().toISOString()
      };
      
      this.io.to(room).emit(eventType, eventData);
      
      const roomSize = this.io.sockets.adapter.rooms.get(room)?.size || 0;
      logger.debug(`Broadcasted ${eventType} to org ${organizationId} (${roomSize} clients)`);
    } catch (error: any) {
      logger.error(`Failed to broadcast ${eventType} to org ${organizationId}:`, error.message);
    }
  }
  
  getConnectedClientsCount(): number {
    return this.connectedClients.size;
  }
}
