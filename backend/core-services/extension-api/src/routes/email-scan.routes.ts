import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { EmailScannerService } from '../services/email-scanner.service';
import { ExtensionRequest } from '../middleware/extension-auth.middleware';
import { logger } from '../utils/logger';

const router = Router();

// Validation schema
const emailScanSchema = z.object({
  emailContent: z.string().min(1, 'Email content is required'),
  privacyMode: z.boolean().optional(),
  scanLinks: z.boolean().optional(),
  includeFullAnalysis: z.boolean().optional(),
});

/**
 * POST /api/v1/extension/scan-email
 * Scan email content for phishing threats
 */
router.post('/', async (req: ExtensionRequest, res: Response) => {
  try {
    // Validate request body
    const validationResult = emailScanSchema.safeParse(req.body);
    
    if (!validationResult.success) {
      logger.warn('Invalid email scan request', {
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
    
    const { emailContent, privacyMode, scanLinks, includeFullAnalysis } = validationResult.data;
    
    // Get email scanner service from app context
    const emailScanner = req.app.get('emailScanner') as EmailScannerService;
    
    if (!emailScanner) {
      logger.error('Email scanner service not available');
      return res.status(500).json({
        error: {
          message: 'Service unavailable',
          statusCode: 500
        }
      });
    }
    
    // Scan email
    const result = await emailScanner.scanEmail(emailContent, {
      privacyMode: privacyMode || false,
      scanLinks: scanLinks || false,
      includeFullAnalysis: includeFullAnalysis || false
    });
    
    logger.info('Email scan completed', {
      isThreat: result.isThreat,
      suspiciousLinksCount: result.suspiciousLinks?.length || 0,
      extensionId: req.extensionId
    });
    
    res.json(result);
  } catch (error: any) {
    logger.error('Email scan route error', {
      error: error.message,
      stack: error.stack,
      extensionId: req.extensionId
    });
    
    res.status(500).json({
      error: {
        message: 'Email scan failed',
        statusCode: 500
      }
    });
  }
});

export default router;
