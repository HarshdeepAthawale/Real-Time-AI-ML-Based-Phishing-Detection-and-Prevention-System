import { Router, Request, Response } from 'express';
import { AnalyticsService } from '../services/analytics.service';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.middleware';
import { logger } from '../utils/logger';

const router = Router();
const analyticsService = new AnalyticsService();

/**
 * Get timeline data for charts
 */
router.get('/timeline',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { start_date, end_date, interval = 'day' } = req.query;
      const organizationId = req.user?.organizationId;

      if (!organizationId) {
        return res.status(400).json({ error: 'Organization ID required' });
      }

      const startDate = start_date ? new Date(start_date as string) : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
      const endDate = end_date ? new Date(end_date as string) : new Date();

      const data = await analyticsService.getTimelineData(
        organizationId,
        startDate,
        endDate,
        interval as any
      );

      res.json(data);
    } catch (error: any) {
      logger.error(`Timeline data error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Get threat distribution
 */
router.get('/distribution',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const organizationId = req.user?.organizationId;

      if (!organizationId) {
        return res.status(400).json({ error: 'Organization ID required' });
      }

      const data = await analyticsService.getThreatDistribution(organizationId);

      res.json(data);
    } catch (error: any) {
      logger.error(`Distribution data error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Get performance metrics
 */
router.get('/performance',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const organizationId = req.user?.organizationId;

      if (!organizationId) {
        return res.status(400).json({ error: 'Organization ID required' });
      }

      const data = await analyticsService.getPerformanceMetrics(organizationId);

      res.json(data);
    } catch (error: any) {
      logger.error(`Performance metrics error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Get trend analysis
 */
router.get('/trends',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { days = 30 } = req.query;
      const organizationId = req.user?.organizationId;

      if (!organizationId) {
        return res.status(400).json({ error: 'Organization ID required' });
      }

      const data = await analyticsService.getTrendAnalysis(
        organizationId,
        parseInt(days as string)
      );

      res.json(data);
    } catch (error: any) {
      logger.error(`Trend analysis error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Get top threats
 */
router.get('/top-threats',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { limit = 10 } = req.query;
      const organizationId = req.user?.organizationId;

      if (!organizationId) {
        return res.status(400).json({ error: 'Organization ID required' });
      }

      const data = await analyticsService.getTopThreats(
        organizationId,
        parseInt(limit as string)
      );

      res.json(data);
    } catch (error: any) {
      logger.error(`Top threats error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Get detection accuracy metrics
 */
router.get('/accuracy',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const organizationId = req.user?.organizationId;

      if (!organizationId) {
        return res.status(400).json({ error: 'Organization ID required' });
      }

      const data = await analyticsService.getDetectionAccuracy(organizationId);

      res.json(data);
    } catch (error: any) {
      logger.error(`Accuracy metrics error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Get hourly distribution (heatmap data)
 */
router.get('/hourly-distribution',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { days = 7 } = req.query;
      const organizationId = req.user?.organizationId;

      if (!organizationId) {
        return res.status(400).json({ error: 'Organization ID required' });
      }

      const data = await analyticsService.getHourlyDistribution(
        organizationId,
        parseInt(days as string)
      );

      res.json(data);
    } catch (error: any) {
      logger.error(`Hourly distribution error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Get ML model performance comparison
 */
router.get('/ml-performance',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const organizationId = req.user?.organizationId;

      if (!organizationId) {
        return res.status(400).json({ error: 'Organization ID required' });
      }

      const data = await analyticsService.getMLModelPerformance(organizationId);

      res.json(data);
    } catch (error: any) {
      logger.error(`ML performance error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

export default router;
