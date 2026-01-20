import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import { config } from './config';
import { logger } from './utils/logger';
import { errorHandler } from './middleware/error-handler.middleware';
import detectionRoutes from './routes/detection.routes';
import { setupWebSocket } from './routes/websocket.routes';
import { CacheService } from './services/cache.service';
import { EventStreamerService } from './services/event-streamer.service';
import { setEventStreamer } from './routes/detection.routes';

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: config.cors.origin,
    methods: ['GET', 'POST']
  }
});

// Middleware
app.use(helmet());
app.use(cors(config.cors));
app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Initialize services
const cacheService = new CacheService();
const eventStreamer = new EventStreamerService(io);

// Set event streamer in routes
setEventStreamer(eventStreamer);

// Routes
app.use('/api/v1/detect', detectionRoutes);

// WebSocket setup
setupWebSocket(io, eventStreamer);

// Health check
app.get('/health', async (req, res) => {
  const cacheStatus = await cacheService.isConnected();
  const wsClients = eventStreamer.getConnectedClientsCount();
  
  res.json({
    status: 'healthy',
    service: 'detection-api',
    cache: cacheStatus ? 'connected' : 'disconnected',
    websocket: {
      connectedClients: wsClients
    },
    timestamp: new Date().toISOString()
  });
});

// Error handling
app.use(errorHandler);

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  await cacheService.disconnect();
  httpServer.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully');
  await cacheService.disconnect();
  httpServer.close(() => {
    logger.info('HTTP server closed');
    process.exit(0);
  });
});

// Start server
const PORT = config.port;
httpServer.listen(PORT, () => {
  logger.info(`Detection API server running on port ${PORT}`);
  logger.info(`WebSocket server ready for connections`);
});
