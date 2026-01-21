import { Router, Request, Response } from 'express';
import multer from 'multer';
import { SandboxSubmitterService } from '../services/sandbox-submitter.service';
import { ResultProcessorService } from '../services/result-processor.service';
import { logger } from '../utils/logger';
import { SandboxQueueJob } from '../jobs/sandbox-queue.job';
import { DataSource } from 'typeorm';
import { SandboxAnalysis } from '../../../../shared/database/models/SandboxAnalysis';

const router = Router();

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  },
});

// Middleware to extract organization ID (would come from auth middleware in production)
const getOrganizationId = (req: Request): string => {
  // In production, this would come from JWT token or API key
  return req.headers['x-organization-id'] as string || 
         (req.body.organization_id as string) ||
         '00000000-0000-0000-0000-000000000000'; // Default org ID
};

export function setupSandboxRoutes(
  submitterService: SandboxSubmitterService,
  resultProcessorService: ResultProcessorService,
  queueJob: SandboxQueueJob,
  dataSource: DataSource
): Router {
  const sandboxRepository = dataSource.getRepository(SandboxAnalysis);

  /**
   * POST /api/v1/sandbox/analyze/file
   * Submit a file for sandbox analysis
   */
  router.post('/analyze/file', upload.single('file'), async (req: Request, res: Response) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          error: 'No file provided',
          message: 'Please provide a file in the request'
        });
      }

      const organizationId = getOrganizationId(req);
      const filename = req.file.originalname || 'unknown';
      
      logger.info('File submission request', { filename, size: req.file.size, organizationId });

      const analysisId = await submitterService.submitFile(
        req.file.buffer,
        filename,
        organizationId
      );

      // Add job to queue for result processing (if sandbox submission was successful)
      const analysis = await sandboxRepository.findOne({ where: { id: analysisId } });
      if (analysis?.sandbox_job_id) {
        // Poll after initial delay
        await queueJob.addAnalysisJob(analysisId, 30); // Start polling after 30 seconds
      }

      res.status(202).json({
        analysis_id: analysisId,
        status: 'pending',
        message: 'File submitted for sandbox analysis'
      });
    } catch (error: any) {
      logger.error('File submission error', { error: error.message });
      res.status(500).json({
        error: 'File submission failed',
        message: error.message
      });
    }
  });

  /**
   * POST /api/v1/sandbox/analyze/url
   * Submit a URL for sandbox analysis
   */
  router.post('/analyze/url', async (req: Request, res: Response) => {
    try {
      const { url } = req.body;
      
      if (!url || typeof url !== 'string') {
        return res.status(400).json({
          error: 'Invalid URL',
          message: 'Please provide a valid URL'
        });
      }

      // Basic URL validation
      try {
        new URL(url);
      } catch {
        return res.status(400).json({
          error: 'Invalid URL format',
          message: 'Please provide a valid URL'
        });
      }

      const organizationId = getOrganizationId(req);
      
      logger.info('URL submission request', { url, organizationId });

      const analysisId = await submitterService.submitURL(url, organizationId);

      // Add job to queue for result processing
      await queueJob.addAnalysisJob(analysisId, 30); // Start polling after 30 seconds

      res.status(202).json({
        analysis_id: analysisId,
        status: 'pending',
        message: 'URL submitted for sandbox analysis'
      });
    } catch (error: any) {
      logger.error('URL submission error', { error: error.message });
      res.status(500).json({
        error: 'URL submission failed',
        message: error.message
      });
    }
  });

  /**
   * GET /api/v1/sandbox/analysis/:id
   * Get analysis status and results
   */
  router.get('/analysis/:id', async (req: Request, res: Response) => {
    try {
      const { id } = req.params;
      
      const analysis = await sandboxRepository.findOne({
        where: { id },
        relations: ['threat']
      });

      if (!analysis) {
        return res.status(404).json({
          error: 'Analysis not found',
          message: `Analysis with ID ${id} not found`
        });
      }

      // Format response
      const response: any = {
        analysis_id: analysis.id,
        status: analysis.status,
        analysis_type: analysis.analysis_type,
        submitted_at: analysis.submitted_at,
        started_at: analysis.started_at,
        completed_at: analysis.completed_at,
        sandbox_provider: analysis.sandbox_provider,
        sandbox_job_id: analysis.sandbox_job_id,
      };

      if (analysis.result_data) {
        response.results = analysis.result_data;
      }

      if (analysis.threat_id) {
        response.threat_id = analysis.threat_id;
        if (analysis.threat) {
          response.threat = {
            id: analysis.threat.id,
            threat_type: analysis.threat.threat_type,
            severity: analysis.threat.severity,
            confidence_score: analysis.threat.confidence_score,
          };
        }
      }

      res.json(response);
    } catch (error: any) {
      logger.error('Get analysis error', { error: error.message, id: req.params.id });
      res.status(500).json({
        error: 'Failed to get analysis',
        message: error.message
      });
    }
  });

  /**
   * GET /api/v1/sandbox/analyses
   * List analyses with pagination
   */
  router.get('/analyses', async (req: Request, res: Response) => {
    try {
      const page = parseInt(req.query.page as string) || 1;
      const limit = parseInt(req.query.limit as string) || 20;
      const skip = (page - 1) * limit;
      
      const organizationId = getOrganizationId(req);
      
      const [analyses, total] = await sandboxRepository.findAndCount({
        where: { organization_id: organizationId },
        order: { submitted_at: 'DESC' },
        take: limit,
        skip,
      });

      res.json({
        analyses: analyses.map(a => ({
          analysis_id: a.id,
          status: a.status,
          analysis_type: a.analysis_type,
          submitted_at: a.submitted_at,
          completed_at: a.completed_at,
          threat_id: a.threat_id,
        })),
        pagination: {
          page,
          limit,
          total,
          total_pages: Math.ceil(total / limit),
        }
      });
    } catch (error: any) {
      logger.error('List analyses error', { error: error.message });
      res.status(500).json({
        error: 'Failed to list analyses',
        message: error.message
      });
    }
  });

  return router;
}
