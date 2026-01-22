import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { errorHandler } from './middleware/errorHandler';
import { requestLogger } from './middleware/requestLogger';
import { apiKeyAuth } from './middleware/apiKeyAuth';
import { rateLimiter } from './middleware/rateLimiter';
import { setupRoutes } from './routes';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.CORS_ORIGIN || '*',
  credentials: true
}));

// Body parsing
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging
app.use(requestLogger);

// Health check endpoint (no auth required)
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// WebSocket routes (Socket.io) - no auth middleware as Socket.io handles auth in handshake
// Socket.io connections go directly to detection-api service
const detectionApiUrl = process.env.DETECTION_API_URL || 'http://detection-api:3001';
app.use('/socket.io', createProxyMiddleware({
  target: detectionApiUrl,
  changeOrigin: true,
  ws: true, // Enable WebSocket support
  logLevel: process.env.NODE_ENV === 'development' ? 'debug' : 'warn',
  onProxyReq: (proxyReq: any, req: any) => {
    // Forward authentication headers for WebSocket upgrade
    const apiKey = req.headers['x-api-key'];
    const orgId = req.headers['x-organization-id'];
    if (apiKey) {
      proxyReq.setHeader('X-API-Key', apiKey);
    }
    if (orgId) {
      proxyReq.setHeader('X-Organization-ID', orgId);
    }
  }
}));

// API routes
app.use('/api/v1', apiKeyAuth, rateLimiter, setupRoutes());

// Error handling
app.use(errorHandler);

app.listen(PORT, () => {
  console.log(`API Gateway running on port ${PORT}`);
});
