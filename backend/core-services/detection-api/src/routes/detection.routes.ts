import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { OrchestratorService } from '../services/orchestrator.service';
import { DecisionEngineService } from '../services/decision-engine.service';
import { CacheService } from '../services/cache.service';
import { EventStreamerService } from '../services/event-streamer.service';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.middleware';
import { rateLimitMiddleware } from '../middleware/rate-limit.middleware';
import { detectEmailSchema, detectURLSchema, detectTextSchema } from '../utils/validators';
import { logger } from '../utils/logger';

const router = Router();
const orchestrator = new OrchestratorService();
const decisionEngine = new DecisionEngineService();
const cacheService = new CacheService();
let eventStreamer: EventStreamerService;

export function setEventStreamer(streamer: EventStreamerService): void {
  eventStreamer = streamer;
}

router.post('/email', 
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const validated = detectEmailSchema.parse(req.body);
      
      // Check cache
      const cacheKey = cacheService.generateCacheKey('email', validated.emailContent);
      const cached = await cacheService.get(cacheKey);
      
      if (cached) {
        logger.debug('Cache hit for email detection', { cacheKey });
        return res.json({
          ...cached,
          cached: true
        });
      }
      
      // Analyze
      const mlResponse = await orchestrator.analyzeEmail({
        emailContent: validated.emailContent,
        includeFeatures: validated.includeFeatures,
        organizationId: validated.organizationId || req.organizationId
      });
      
      const threat = decisionEngine.makeDecision(mlResponse, validated);
      
      // Cache result
      await cacheService.set(cacheKey, threat, 3600);
      
      // Broadcast event
      if (eventStreamer && (validated.organizationId || req.organizationId)) {
        eventStreamer.broadcastThreat(
          validated.organizationId || req.organizationId || '',
          threat
        );
      }
      
      res.json(threat);
    } catch (error) {
      if (error instanceof z.ZodError) {
        logger.warn('Validation error', { errors: error.errors });
        return res.status(400).json({ 
          error: 'Validation failed',
          details: error.errors 
        });
      }
      logger.error('Email detection error', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

router.post('/url',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const validated = detectURLSchema.parse(req.body);
      
      // Check cache
      const cacheKey = cacheService.generateCacheKey('url', validated.url);
      const cached = await cacheService.get(cacheKey);
      
      if (cached) {
        logger.debug('Cache hit for URL detection', { cacheKey });
        return res.json({
          ...cached,
          cached: true
        });
      }
      
      // Analyze
      const mlResponse = await orchestrator.analyzeURL({
        url: validated.url,
        legitimateDomain: validated.legitimateDomain,
        legitimateUrl: validated.legitimateUrl,
        organizationId: validated.organizationId || req.organizationId
      });
      
      const threat = decisionEngine.makeDecision(mlResponse, validated);
      
      // Cache result (URLs cached longer)
      await cacheService.set(cacheKey, threat, 7200);
      
      // Broadcast event
      if (eventStreamer) {
        eventStreamer.broadcastEvent('url_analyzed', {
          url: validated.url,
          threat
        });
      }
      
      res.json(threat);
    } catch (error) {
      if (error instanceof z.ZodError) {
        logger.warn('Validation error', { errors: error.errors });
        return res.status(400).json({ 
          error: 'Validation failed',
          details: error.errors 
        });
      }
      logger.error('URL detection error', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

router.post('/text',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const validated = detectTextSchema.parse(req.body);
      
      // Check cache
      const cacheKey = cacheService.generateCacheKey('text', validated.text);
      const cached = await cacheService.get(cacheKey);
      
      if (cached) {
        logger.debug('Cache hit for text detection', { cacheKey });
        return res.json({
          ...cached,
          cached: true
        });
      }
      
      // Analyze
      const mlResponse = await orchestrator.analyzeText({
        text: validated.text,
        includeFeatures: validated.includeFeatures,
        organizationId: validated.organizationId || req.organizationId
      });
      
      const threat = decisionEngine.makeDecision(mlResponse, validated);
      
      // Cache result
      await cacheService.set(cacheKey, threat, 3600);
      
      res.json(threat);
    } catch (error) {
      if (error instanceof z.ZodError) {
        logger.warn('Validation error', { errors: error.errors });
        return res.status(400).json({ 
          error: 'Validation failed',
          details: error.errors 
        });
      }
      logger.error('Text detection error', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

export default router;
