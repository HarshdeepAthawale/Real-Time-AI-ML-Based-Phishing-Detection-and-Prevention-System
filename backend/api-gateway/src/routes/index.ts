import { Router } from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { serviceConfigs, routeConfig } from '../config/gateway';

export const setupRoutes = (): Router => {
  const router = Router();

  // Create proxy middleware for each service
  Object.entries(routeConfig).forEach(([path, serviceName]) => {
    const serviceConfig = serviceConfigs[serviceName];
    
    if (serviceConfig) {
      router.use(
        path,
        createProxyMiddleware({
          target: serviceConfig.url,
          changeOrigin: true,
          timeout: serviceConfig.timeout,
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
            proxyReq.setHeader('X-Forwarded-For', req.ip);
            proxyReq.setHeader('X-Original-Path', req.path);
          }
        })
      );
    }
  });

  return router;
};
