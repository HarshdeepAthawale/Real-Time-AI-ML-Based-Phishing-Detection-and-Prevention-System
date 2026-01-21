import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { SyncService } from '../services/sync.service';
import { validateParams } from '../middleware/validation.middleware';
import { NotFoundError } from '../middleware/error-handler.middleware';
import { logger } from '../utils/logger';

const router = Router();

const feedIdSchema = z.object({
  feedId: z.string().uuid(),
});

// Sync all feeds
router.post('/all', async (req: Request, res: Response) => {
  try {
    const syncService = req.app.get('syncService') as SyncService;
    const results = await syncService.syncAllFeeds();
    
    logger.info(`Sync all feeds completed: ${results.length} feeds`);
    
    res.json({
      success: true,
      results,
      summary: {
        total: results.length,
        successful: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length,
        totalIOCsInserted: results.reduce((sum, r) => sum + r.iocsInserted, 0),
      },
    });
  } catch (error) {
    logger.error('Sync all feeds error', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Sync specific feed
router.post('/:feedId', validateParams(feedIdSchema), async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { feedId } = req.params;
    
    const syncService = req.app.get('syncService') as SyncService;
    const result = await syncService.syncFeed(feedId);
    
    logger.info(`Sync feed completed: ${result.feedName} (${feedId})`);
    
    res.json({
      success: result.success,
      result,
    });
  } catch (error) {
    const errorMessage = (error as Error).message || String(error);
    if (errorMessage.includes('not found')) {
      return next(new NotFoundError(errorMessage));
    }
    next(error);
  }
});

// Get sync status
router.get('/status', async (req: Request, res: Response) => {
  try {
    const syncService = req.app.get('syncService') as SyncService;
    const status = await syncService.getSyncStatus();
    
    res.json(status);
  } catch (error) {
    logger.error('Get sync status error', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;
