import { Router, Request, Response } from 'express';
import { DashboardService } from '../services/dashboard.service';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.middleware';
import { rateLimitMiddleware } from '../middleware/rate-limit.middleware';
import { logger } from '../utils/logger';

const router = Router();
const dashboardService = new DashboardService();

/**
 * GET /api/v1/dashboard/stats
 * Get dashboard statistics
 */
router.get('/stats',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const stats = await dashboardService.getStats();
      res.json(stats);
    } catch (error) {
      logger.error('Error fetching dashboard stats', error);
      res.status(500).json({ error: 'Failed to fetch dashboard statistics' });
    }
  }
);

/**
 * GET /api/v1/dashboard/threats
 * Get recent threats
 */
router.get('/threats',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const limit = parseInt(req.query.limit as string) || 10;
      const offset = parseInt(req.query.offset as string) || 0;

      if (limit > 100) {
        return res.status(400).json({ error: 'Limit cannot exceed 100' });
      }

      const threats = await dashboardService.getRecentThreats(limit, offset);
      res.json(threats);
    } catch (error) {
      logger.error('Error fetching recent threats', error);
      res.status(500).json({ error: 'Failed to fetch recent threats' });
    }
  }
);

/**
 * GET /api/v1/dashboard/chart
 * Get chart data for threat timeline
 */
router.get('/chart',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const hours = parseInt(req.query.hours as string) || 24;

      if (hours > 168) { // Max 7 days
        return res.status(400).json({ error: 'Hours cannot exceed 168 (7 days)' });
      }

      const chartData = await dashboardService.getChartData(hours);
      res.json(chartData);
    } catch (error) {
      logger.error('Error fetching chart data', error);
      res.status(500).json({ error: 'Failed to fetch chart data' });
    }
  }
);

/**
 * GET /api/v1/dashboard/distribution
 * Get threat type distribution
 */
router.get('/distribution',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const distribution = await dashboardService.getThreatDistribution();
      res.json(distribution);
    } catch (error) {
      logger.error('Error fetching threat distribution', error);
      res.status(500).json({ error: 'Failed to fetch threat distribution' });
    }
  }
);

export default router;
