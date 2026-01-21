import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { IOCMatcherService } from '../services/ioc-matcher.service';
import { IOCManagerService } from '../services/ioc-manager.service';
import { EnrichmentService } from '../services/enrichment.service';
import { 
  IOCCheckRequest, 
  IOCCheckResponse, 
  IOCBulkCheckRequest,
  IOCBulkCheckResponse,
  IOCType,
  Severity 
} from '../models/ioc.model';
import { validateBody, validateQuery } from '../middleware/validation.middleware';
import { logger } from '../utils/logger';

const router = Router();

const checkIOCSchema = z.object({
  iocType: z.enum(['url', 'domain', 'ip', 'hash_md5', 'hash_sha1', 'hash_sha256', 'filename', 'email']),
  iocValue: z.string().min(1),
});

const bulkCheckIOCSchema = z.object({
  iocs: z.array(checkIOCSchema).min(1).max(100), // Max 100 at a time
});

const searchIOCSchema = z.object({
  iocType: z.enum(['url', 'domain', 'ip', 'hash_md5', 'hash_sha1', 'hash_sha256', 'filename', 'email']).optional(),
  severity: z.enum(['critical', 'high', 'medium', 'low']).optional(),
  threatType: z.string().optional(),
  search: z.string().optional(),
  limit: z.coerce.number().int().min(1).max(100).optional(),
  offset: z.coerce.number().int().min(0).optional(),
});

const reportIOCSchema = z.object({
  iocType: z.enum(['url', 'domain', 'ip', 'hash_md5', 'hash_sha1', 'hash_sha256', 'filename', 'email']),
  iocValue: z.string().min(1),
  threatType: z.string().optional(),
  severity: z.enum(['critical', 'high', 'medium', 'low']).optional(),
  confidence: z.number().min(0).max(100).optional(),
  metadata: z.record(z.any()).optional(),
});

// Check single IOC
router.post('/check', validateBody(checkIOCSchema), async (req: Request, res: Response) => {
  try {
    const { iocType, iocValue } = req.body as IOCCheckRequest;
    const enrich = req.query.enrich === 'true';
    
    const iocMatcher = req.app.get('iocMatcher') as IOCMatcherService;
    const match = await iocMatcher.matchIOC(iocType, iocValue);
    
    const response: IOCCheckResponse & { enrichment?: any } = {
      found: !!match,
      ioc: match || undefined,
      confidence: match?.confidence || undefined,
    };
    
    // Enrich if requested and IOC found
    if (enrich && match) {
      const enrichmentService = req.app.get('enrichmentService') as EnrichmentService;
      const enrichment = await enrichmentService.enrichIOC(match);
      if (enrichment.enriched) {
        response.enrichment = enrichment.additionalContext;
      }
    }
    
    res.json(response);
  } catch (error) {
    logger.error('IOC check error', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Bulk check IOCs
router.post('/bulk-check', validateBody(bulkCheckIOCSchema), async (req: Request, res: Response) => {
  try {
    const { iocs } = req.body as IOCBulkCheckRequest;
    
    const iocMatcher = req.app.get('iocMatcher') as IOCMatcherService;
    const results: IOCCheckResponse[] = [];
    
    for (const ioc of iocs) {
      try {
        const match = await iocMatcher.matchIOC(ioc.iocType, ioc.iocValue);
        results.push({
          found: !!match,
          ioc: match || undefined,
          confidence: match?.confidence || undefined,
        });
      } catch (error) {
        logger.error(`Bulk check error for IOC: ${ioc.iocValue}`, error);
        results.push({
          found: false,
        });
      }
    }
    
    const summary = {
      total: results.length,
      found: results.filter(r => r.found).length,
      notFound: results.filter(r => !r.found).length,
    };
    
    const response: IOCBulkCheckResponse = {
      results,
      summary,
    };
    
    res.json(response);
  } catch (error) {
    logger.error('Bulk IOC check error', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Search IOCs
router.get('/search', validateQuery(searchIOCSchema), async (req: Request, res: Response) => {
  try {
    const params = req.query as any;
    
    const iocManager = req.app.get('iocManager') as IOCManagerService;
    const { iocs, total } = await iocManager.searchIOCs({
      iocType: params.iocType as IOCType | undefined,
      severity: params.severity as Severity | undefined,
      threatType: params.threatType,
      search: params.search,
      limit: params.limit ? parseInt(params.limit, 10) : undefined,
      offset: params.offset ? parseInt(params.offset, 10) : undefined,
    });
    
    res.json({
      iocs,
      total,
      limit: params.limit || 50,
      offset: params.offset || 0,
    });
  } catch (error) {
    logger.error('IOC search error', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Report new IOC
router.post('/report', validateBody(reportIOCSchema), async (req: Request, res: Response) => {
  try {
    const data = req.body as any;
    
    const iocManager = req.app.get('iocManager') as IOCManagerService;
    const iocMatcher = req.app.get('iocMatcher') as IOCMatcherService;
    
    const ioc = await iocManager.createIOC({
      iocType: data.iocType,
      iocValue: data.iocValue,
      threatType: data.threatType,
      severity: data.severity || 'medium',
      confidence: data.confidence || 70,
      metadata: data.metadata || {},
      source: 'user',
    });
    
    // Add to bloom filter
    await iocMatcher.addToBloomFilter(ioc.iocType, ioc.iocValue);
    
    logger.info(`IOC reported: ${ioc.iocValue} (${ioc.iocType})`);
    
    res.status(201).json(ioc);
  } catch (error) {
    logger.error('IOC report error', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get IOC statistics
router.get('/stats', async (req: Request, res: Response) => {
  try {
    const iocManager = req.app.get('iocManager') as IOCManagerService;
    const stats = await iocManager.getStats();
    
    res.json(stats);
  } catch (error) {
    logger.error('IOC stats error', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;
