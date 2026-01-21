import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { URLCheckerService } from '../services/url-checker.service';
import { ExtensionRequest } from '../middleware/extension-auth.middleware';
import { logger } from '../utils/logger';

const router = Router();

// Validation schema
const urlCheckSchema = z.object({
  url: z.string().url('Invalid URL format'),
  includeFullAnalysis: z.boolean().optional(),
  privacyMode: z.boolean().optional(),
  pageText: z.string().optional(),
  pageTitle: z.string().optional(),
  screenshot: z.string().optional(), // base64 encoded
});

/**
 * POST /api/v1/extension/check-url
 * Check if a URL is a phishing threat
 */
router.post('/', async (req: ExtensionRequest, res: Response) => {
  try {
    // Validate request body
    const validationResult = urlCheckSchema.safeParse(req.body);
    
    if (!validationResult.success) {
      logger.warn('Invalid URL check request', {
        errors: validationResult.error.errors,
        extensionId: req.extensionId
      });
      
      return res.status(400).json({
        error: {
          message: 'Validation failed',
          statusCode: 400,
          details: validationResult.error.errors
        }
      });
    }
    
    const { url, includeFullAnalysis, privacyMode, pageText, pageTitle, screenshot } = validationResult.data;
    
    // Get URL checker service from app context
    const urlChecker = req.app.get('urlChecker') as URLCheckerService;
    
    if (!urlChecker) {
      logger.error('URL checker service not available');
      return res.status(500).json({
        error: {
          message: 'Service unavailable',
          statusCode: 500
        }
      });
    }
    
    // Check URL
    const result = await urlChecker.checkURL(url, {
      includeFullAnalysis: includeFullAnalysis || false,
      privacyMode: privacyMode || false,
      pageText,
      pageTitle,
      screenshot
    });
    
    logger.info('URL check completed', {
      url: url.substring(0, 100),
      isThreat: result.isThreat,
      cached: result.cached,
      extensionId: req.extensionId
    });
    
    res.json(result);
  } catch (error: any) {
    logger.error('URL check route error', {
      error: error.message,
      stack: error.stack,
      extensionId: req.extensionId
    });
    
    res.status(500).json({
      error: {
        message: 'URL check failed',
        statusCode: 500
      }
    });
  }
});

export default router;
