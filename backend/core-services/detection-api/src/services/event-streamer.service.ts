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
    this.io.to(`org:${organizationId}`).emit('threat_detected', {
      ...threat,
      timestamp: new Date().toISOString()
    });
  }
  
  broadcastEvent(eventType: string, data: any): void {
    this.io.emit(eventType, {
      ...data,
      timestamp: new Date().toISOString()
    });
  }
  
  getConnectedClientsCount(): number {
    return this.connectedClients.size;
  }
}
