import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { OrchestratorService } from '../services/orchestrator.service';
import { DecisionEngineService } from '../services/decision-engine.service';
import { CacheService } from '../services/cache.service';
import { EventStreamerService } from '../services/event-streamer.service';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.middleware';
import { rateLimitMiddleware } from '../middleware/rate-limit.middleware';
import { detectEmailSchema, detectURLSchema, detectTextSchema, detectSMSSchema } from '../utils/validators';
import { logger } from '../utils/logger';
import { getPostgreSQL } from '../../../../shared/database/connection';
import { Threat as ThreatEntity } from '../../../../shared/database/models/Threat';
import { Detection as DetectionEntity } from '../../../../shared/database/models/Detection';
import { Threat as ThreatModel } from '../models/detection.model';

const router = Router();
const orchestrator = new OrchestratorService();
const decisionEngine = new DecisionEngineService();
const cacheService = new CacheService();
let eventStreamer: EventStreamerService;

export function setEventStreamer(streamer: EventStreamerService): void {
  eventStreamer = streamer;
}

/**
 * Save threat to database
 */
async function saveThreatToDatabase(
  threat: ThreatModel,
  organizationId: string | undefined,
  input: any,
  mlResponse: any
): Promise<string | null> {
  try {
    if (!organizationId) {
      logger.debug('Skipping threat save: no organization ID');
      return null;
    }

    const dataSource = getPostgreSQL();
    const threatRepository = dataSource.getRepository(ThreatEntity);
    const detectionRepository = dataSource.getRepository(DetectionEntity);

    // Only save if it's actually a threat
    if (!threat.isThreat) {
      return null;
    }

    // Create threat entity
    const threatEntity = threatRepository.create({
      organization_id: organizationId,
      threat_type: threat.threatType,
      severity: threat.severity,
      status: 'detected',
      confidence_score: Math.round(threat.confidence * 100) / 100,
      source: input.url ? 'url' : input.emailContent ? 'email' : 'text',
      source_value: input.url || input.emailContent?.substring(0, 500) || input.text?.substring(0, 500) || null,
      title: threat.indicators.length > 0 ? threat.indicators[0] : null,
      description: threat.indicators.join(', '),
      metadata: {
        ...threat.metadata,
        scores: threat.scores,
        indicators: threat.indicators,
      },
      detected_at: new Date(),
    });

    const savedThreat = await threatRepository.save(threatEntity);

    // Create detection record
    const detectionEntity = detectionRepository.create({
      threat_id: savedThreat.id,
      organization_id: organizationId,
      detection_type: 'ensemble',
      model_version: '1.0.0',
      input_data: {
        type: input.url ? 'url' : input.emailContent ? 'email' : 'text',
        value: input.url || input.emailContent?.substring(0, 1000) || input.text?.substring(0, 1000),
      },
      analysis_result: mlResponse,
      confidence_score: Math.round(threat.confidence * 100) / 100,
      processing_time_ms: threat.metadata.processingTimeMs,
      detected_at: new Date(),
    });

    await detectionRepository.save(detectionEntity);

    logger.info('Threat saved to database', {
      threatId: savedThreat.id,
      organizationId,
      threatType: threat.threatType,
      severity: threat.severity,
    });

    return savedThreat.id;
  } catch (error) {
    logger.error('Failed to save threat to database', error);
    // Don't throw - allow detection to continue even if save fails
    return null;
  }
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
      
      // Save threat to database if detected
      const orgId = validated.organizationId || req.organizationId;
      const threatId = await saveThreatToDatabase(threat, orgId, validated, mlResponse);
      
      // Cache result
      await cacheService.set(cacheKey, threat, 3600);
      
      // Broadcast event if threat detected
      if (eventStreamer && threat.isThreat) {
        if (orgId) {
          eventStreamer.broadcastThreat(orgId, threat);
        } else {
          // Broadcast to all if no org ID
          eventStreamer.broadcastEvent('threat_detected', {
            type: 'email',
            threat
          });
        }
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
      
      // Save threat to database if detected
      const orgId = validated.organizationId || req.organizationId;
      const threatId = await saveThreatToDatabase(threat, orgId, validated, mlResponse);
      
      // Cache result (URLs cached longer)
      await cacheService.set(cacheKey, threat, 7200);
      
      // Broadcast event if threat detected
      if (eventStreamer && threat.isThreat) {
        if (orgId) {
          eventStreamer.broadcastThreat(orgId, threat);
        } else {
          // Broadcast to all if no org ID
          eventStreamer.broadcastEvent('threat_detected', {
            type: 'url',
            url: validated.url,
            threat
          });
        }
      } else if (eventStreamer) {
        // Broadcast analysis completion even if not a threat
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
      
      // Save threat to database if detected
      const orgId = validated.organizationId || req.organizationId;
      const threatId = await saveThreatToDatabase(threat, orgId, validated, mlResponse);
      
      // Cache result
      await cacheService.set(cacheKey, threat, 3600);
      
      // Broadcast event if threat detected
      if (eventStreamer && threat.isThreat) {
        if (orgId) {
          eventStreamer.broadcastThreat(orgId, threat);
        } else {
          // Broadcast to all if no org ID (for general alerts)
          eventStreamer.broadcastEvent('threat_detected', {
            type: 'text',
            threat
          });
        }
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
      logger.error('Text detection error', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

router.post('/sms',
  authMiddleware,
  rateLimitMiddleware,
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const validated = detectSMSSchema.parse(req.body);

      // Check cache
      const cacheKey = cacheService.generateCacheKey('sms', validated.message);
      const cached = await cacheService.get(cacheKey);

      if (cached) {
        logger.debug('Cache hit for SMS detection', { cacheKey });
        return res.json({
          ...cached,
          cached: true
        });
      }

      // Extract URLs from SMS for URL analysis
      const urlPattern = /https?:\/\/[^\s]+/g;
      const urls = validated.message.match(urlPattern) || [];

      // Analyze SMS text using NLP pipeline (no header parsing needed)
      const mlResponse = await orchestrator.analyzeText({
        text: validated.message,
        includeFeatures: false,
        organizationId: validated.organizationId || req.organizationId
      });

      // If URLs found in SMS, also run URL analysis
      if (urls.length > 0) {
        try {
          const urlResponse = await orchestrator.analyzeURL({
            url: urls[0],
            organizationId: validated.organizationId || req.organizationId
          });
          mlResponse.url = urlResponse.url;
        } catch (err) {
          logger.warn('URL analysis in SMS failed', err);
        }
      }

      const threat = decisionEngine.makeDecision(mlResponse, {
        ...validated,
        text: validated.message
      });

      // Add SMS-specific metadata
      (threat as any).inputType = 'sms';
      (threat as any).sender = validated.sender || null;
      (threat as any).urlsFound = urls;

      // Save threat to database if detected
      const orgId = validated.organizationId || req.organizationId;
      const threatId = await saveThreatToDatabase(threat, orgId, validated, mlResponse);

      // Cache result
      await cacheService.set(cacheKey, threat, 3600);

      // Broadcast event if threat detected
      if (eventStreamer && threat.isThreat) {
        if (orgId) {
          eventStreamer.broadcastThreat(orgId, threat);
        } else {
          eventStreamer.broadcastEvent('threat_detected', {
            type: 'sms',
            threat
          });
        }
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
      logger.error('SMS detection error', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

export default router;
