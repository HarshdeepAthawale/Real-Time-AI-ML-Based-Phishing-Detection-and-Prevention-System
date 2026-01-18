import { Router, RequestHandler } from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { serviceConfigs, routeConfig } from '../config/gateway';
import { detectionHandler } from '../handlers/detection';
import { threatIntelHandler } from '../handlers/threatIntel';

export const setupRoutes = (): Router => {
  const router = Router();

  // Create proxy middleware for each service
  Object.entries(routeConfig).forEach(([path, serviceName]) => {
    const serviceConfig = serviceConfigs[serviceName];
    
    if (serviceConfig) {
      // Apply service-specific handlers
      const handlers: RequestHandler[] = [];
      if (serviceName === 'detection-api') {
        handlers.push(detectionHandler);
      } else if (serviceName === 'threat-intel') {
        handlers.push(threatIntelHandler);
      }
      
      router.use(
        path,
        ...handlers,
        createProxyMiddleware({
          target: serviceConfig.url,
          changeOrigin: true,
          timeout: serviceConfig.timeout,
          ws: path === '/ws/events', // Enable WebSocket for events endpoint
          onError: (err, req, res) => {
            console.error(`Proxy error for ${serviceName}:`, err);
            res.status(503).json({
              error: {
                message: `Service ${serviceName} is unavailable`,
                statusCode: 503
              }
            });
          },
          onProxyReq: (proxyReq, req) => {
            // Forward original headers
            proxyReq.setHeader('X-Forwarded-For', req.ip || 'unknown');
            proxyReq.setHeader('X-Original-Path', req.path);
          }
        })
      );
    }
  });

  // WebSocket route for real-time events (if needed)
  // Note: WebSocket support requires additional setup in detection-api service
  router.use(
    '/ws/events',
    detectionHandler,
    createProxyMiddleware({
      target: serviceConfigs['detection-api'].url,
      changeOrigin: true,
      ws: true,
      logLevel: 'debug'
    })
  );

  return router;
};
