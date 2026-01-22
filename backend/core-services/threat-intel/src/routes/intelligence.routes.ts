import { Router, Request, Response } from 'express';
import { IntelligenceService } from '../services/intelligence.service';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.middleware';
import { rateLimitMiddleware } from '../middleware/rate-limit.middleware';
import { logger } from '../utils/logger';

const router = Router();
const intelligenceService = new IntelligenceService();

// Initialize service on startup
intelligenceService.initialize().catch(error => {
  logger.error(`Failed to initialize intelligence service: ${error.message}`);
});

/**
 * Check URL against threat intelligence
 */
router.post('/check/url',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { url } = req.body;

      if (!url) {
        return res.status(400).json({ error: 'URL is required' });
      }

      const result = await intelligenceService.checkURL(url);

      res.json(result);
    } catch (error: any) {
      logger.error(`URL check error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Check domain against threat intelligence
 */
router.post('/check/domain',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { domain } = req.body;

      if (!domain) {
        return res.status(400).json({ error: 'Domain is required' });
      }

      const result = await intelligenceService.checkDomain(domain);

      res.json(result);
    } catch (error: any) {
      logger.error(`Domain check error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Check IP against threat intelligence
 */
router.post('/check/ip',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const { ip } = req.body;

      if (!ip) {
        return res.status(400).json({ error: 'IP address is required' });
      }

      const result = await intelligenceService.checkIP(ip);

      res.json(result);
    } catch (error: any) {
      logger.error(`IP check error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Trigger feed synchronization
 */
router.post('/feeds/sync',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      // Only allow admins to trigger sync
      if (req.user?.role !== 'admin') {
        return res.status(403).json({ error: 'Forbidden: Admin access required' });
      }

      // Start sync in background
      intelligenceService.syncAllFeeds()
        .then(result => {
          logger.info('Feed sync completed:', result);
        })
        .catch(error => {
          logger.error('Feed sync failed:', error);
        });

      res.json({
        message: 'Feed synchronization started',
        status: 'in_progress'
      });
    } catch (error: any) {
      logger.error(`Feed sync trigger error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Get intelligence service statistics
 */
router.get('/stats',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const stats = intelligenceService.getStats();

      res.json({
        ...stats,
        timestamp: new Date().toISOString()
      });
    } catch (error: any) {
      logger.error(`Stats error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

/**
 * Get feed status
 */
router.get('/feeds/status',
  authMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const stats = intelligenceService.getStats();

      res.json({
        feeds: stats.feeds,
        bloom_filters: {
          last_rebuild: stats.bloom_filters.lastRebuildTime,
          needs_rebuild: stats.bloom_filters.needsRebuild,
          total_items: Object.values(stats.bloom_filters.filters)
            .reduce((sum: number, filter: any) => sum + filter.itemCount, 0)
        },
        timestamp: new Date().toISOString()
      });
    } catch (error: any) {
      logger.error(`Feed status error: ${error.message}`);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

export default router;
