import { Router, Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { FeedManagerService } from '../services/feed-manager.service';
import { validateBody, validateParams } from '../middleware/validation.middleware';
import { NotFoundError } from '../middleware/error-handler.middleware';
import { logger } from '../utils/logger';

const router = Router();

const createFeedSchema = z.object({
  name: z.string().min(1),
  feedType: z.enum(['misp', 'otx', 'custom', 'user_submitted']),
  apiEndpoint: z.string().url().optional(),
  apiKeyEncrypted: z.string().optional(),
  syncIntervalMinutes: z.number().int().min(1).max(10080).optional(), // Max 7 days
  isActive: z.boolean().optional(),
});

const updateFeedSchema = createFeedSchema.partial();

const feedIdSchema = z.object({
  id: z.string().uuid(),
});

// List all feeds
router.get('/', async (req: Request, res: Response) => {
  try {
    const feedManager = req.app.get('feedManager') as FeedManagerService;
    const feeds = await feedManager.getAllFeeds();
    
    res.json({ feeds });
  } catch (error) {
    logger.error('Get feeds error', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get feed by ID
router.get('/:id', validateParams(feedIdSchema), async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { id } = req.params;
    
    const feedManager = req.app.get('feedManager') as FeedManagerService;
    const feed = await feedManager.getFeedById(id);
    
    if (!feed) {
      throw new NotFoundError('Feed not found');
    }
    
    res.json(feed);
  } catch (error) {
    next(error);
  }
});

// Create new feed
router.post('/', validateBody(createFeedSchema), async (req: Request, res: Response) => {
  try {
    const feedManager = req.app.get('feedManager') as FeedManagerService;
    const feed = await feedManager.createFeed(req.body);
    
    logger.info(`Feed created: ${feed.name} (${feed.id})`);
    
    res.status(201).json(feed);
  } catch (error) {
    logger.error('Create feed error', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Update feed
router.put('/:id', validateParams(feedIdSchema), validateBody(updateFeedSchema), async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { id } = req.params;
    
    const feedManager = req.app.get('feedManager') as FeedManagerService;
    const feed = await feedManager.updateFeed(id, req.body);
    
    logger.info(`Feed updated: ${feed.name} (${id})`);
    
    res.json(feed);
  } catch (error) {
    const errorMessage = (error as Error).message || String(error);
    if (errorMessage.includes('not found')) {
      return next(new NotFoundError(errorMessage));
    }
    next(error);
  }
});

// Delete feed
router.delete('/:id', validateParams(feedIdSchema), async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { id } = req.params;
    
    const feedManager = req.app.get('feedManager') as FeedManagerService;
    await feedManager.deleteFeed(id);
    
    logger.info(`Feed deleted: ${id}`);
    
    res.status(204).send();
  } catch (error) {
    if ((error as Error).message.includes('not found')) {
      throw new NotFoundError((error as Error).message);
    }
    next(error);
  }
});

// Toggle feed active status
router.post('/:id/toggle', validateParams(feedIdSchema), async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { id } = req.params;
    
    const feedManager = req.app.get('feedManager') as FeedManagerService;
    const feed = await feedManager.toggleFeed(id);
    
    logger.info(`Feed toggled: ${feed.name} (${id}) - Active: ${feed.isActive}`);
    
    res.json(feed);
  } catch (error) {
    if ((error as Error).message.includes('not found')) {
      throw new NotFoundError((error as Error).message);
    }
    next(error);
  }
});

export default router;
