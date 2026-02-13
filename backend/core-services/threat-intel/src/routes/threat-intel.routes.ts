import { Router, Request, Response } from 'express';
import { IOCManagerService } from '../services/ioc-manager.service';
import { FeedManagerService } from '../services/feed-manager.service';
import { logger } from '../utils/logger';

const router = Router();

/**
 * GET /api/v1/intelligence/domains
 * Get malicious domains list
 */
router.get('/domains', async (req: Request, res: Response) => {
  try {
    const iocManager = req.app.get('iocManager') as IOCManagerService;
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;

    // Get domain IOCs
    const result = await iocManager.searchIOCs({
      iocType: 'domain',
      limit,
      offset,
    });
    const iocs = result.iocs;

    // Transform to domain format
    const domains = iocs.map((ioc) => ({
      domain: ioc.iocValue,
      reputation: ioc.severity === 'critical' || ioc.severity === 'high' ? 'Malicious' : 'Suspicious',
      reports: ioc.sourceReports || 0,
      firstSeen: ioc.firstSeenAt?.toISOString(),
      lastSeen: ioc.lastSeenAt?.toISOString(),
    }));

    res.json(domains);
  } catch (error) {
    logger.error('Error fetching malicious domains', error);
    res.status(500).json({ error: 'Failed to fetch malicious domains' });
  }
});

/**
 * GET /api/v1/intelligence/iocs
 * Get IOCs list
 */
router.get('/iocs', async (req: Request, res: Response) => {
  try {
    const iocManager = req.app.get('iocManager') as IOCManagerService;
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;
    const type = req.query.type as string | undefined;

    const searchParams: any = { limit, offset };
    if (type) {
      searchParams.iocType = type;
    }

    const result = await iocManager.searchIOCs(searchParams);
    const iocs = result.iocs;

    // Transform to frontend format
    const formattedIOCs = iocs.map((ioc) => {
      // Map IOC type to display format
      const typeMap: Record<string, string> = {
        'url': 'URL',
        'domain': 'Domain',
        'ip': 'IP Address',
        'email': 'Email',
        'hash_md5': 'File Hash',
        'hash_sha1': 'File Hash',
        'hash_sha256': 'File Hash',
        'filename': 'Filename',
      };
      
      return {
        value: ioc.iocValue,
        type: typeMap[ioc.iocType] || ioc.iocType.charAt(0).toUpperCase() + ioc.iocType.slice(1) as any,
        sources: ioc.sourceReports || 0,
        firstSeen: ioc.firstSeenAt?.toISOString(),
        lastSeen: ioc.lastSeenAt?.toISOString(),
        severity: ioc.severity,
      };
    });

    res.json(formattedIOCs);
  } catch (error) {
    logger.error('Error fetching IOCs', error);
    res.status(500).json({ error: 'Failed to fetch IOCs' });
  }
});

/**
 * GET /api/v1/intelligence/patterns
 * Get threat patterns (aggregated from IOCs)
 */
router.get('/patterns', async (req: Request, res: Response) => {
  try {
    const iocManager = req.app.get('iocManager') as IOCManagerService;
    const limit = parseInt(req.query.limit as string) || 20;

    // Get all IOCs and aggregate by pattern
    const result = await iocManager.searchIOCs({ limit: 1000 });
    const allIOCs = result.iocs;

    // Group by threat type/pattern
    const patternMap = new Map<string, number>();

    allIOCs.forEach((ioc) => {
      const pattern = (ioc.metadata as any)?.pattern || ioc.threatType || 'Unknown Pattern';
      patternMap.set(pattern, (patternMap.get(pattern) || 0) + 1);
    });

    // Convert to array and sort
    const patterns = Array.from(patternMap.entries())
      .map(([pattern, incidents]) => ({
        pattern,
        incidents,
        severity: incidents > 100 ? 'critical' : incidents > 50 ? 'high' : 'medium' as 'critical' | 'high' | 'medium' | 'low',
      }))
      .sort((a, b) => b.incidents - a.incidents)
      .slice(0, limit);

    res.json(patterns);
  } catch (error) {
    logger.error('Error fetching threat patterns', error);
    res.status(500).json({ error: 'Failed to fetch threat patterns' });
  }
});

/**
 * GET /api/v1/intelligence/summary
 * Get threat intelligence summary
 */
router.get('/summary', async (req: Request, res: Response) => {
  try {
    const iocManager = req.app.get('iocManager') as IOCManagerService;
    const feedManager = req.app.get('feedManager') as FeedManagerService;

    const stats = await iocManager.getStats();
    const feeds = await feedManager.getAllFeeds();

    // Calculate zero-day detection rate based on recent IOCs
    const recentCount = stats.recentCount || 0;
    const total = stats.total || 0;
    const zeroDayRate = recentCount > 0 && total > 0
      ? Math.min(100, Math.round((recentCount / total) * 1000))
      : 0;

    res.json({
      knownThreats: total,
      feedIntegrations: feeds.length,
      zeroDayDetection: zeroDayRate,
      lastUpdated: new Date().toISOString(),
    });
  } catch (error) {
    logger.error('Error fetching threat intelligence summary', error);
    res.status(500).json({ error: 'Failed to fetch summary' });
  }
});

export default router;
