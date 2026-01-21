import { Router, Request, Response } from 'express';
import { z } from 'zod';
import axios from 'axios';
import { config } from '../config';
import { ExtensionRequest } from '../middleware/extension-auth.middleware';
import { logger } from '../utils/logger';

const router = Router();

// Validation schema
const reportSchema = z.object({
  url: z.string().url('Invalid URL format').optional(),
  email: z.string().email('Invalid email format').optional(),
  domain: z.string().optional(),
  reason: z.string().min(1, 'Reason is required'),
  description: z.string().optional(),
  threatType: z.string().optional(),
  severity: z.enum(['critical', 'high', 'medium', 'low']).optional(),
  confidence: z.number().min(0).max(100).optional(),
});

/**
 * POST /api/v1/extension/report
 * Report a phishing threat to threat intelligence service
 */
router.post('/', async (req: ExtensionRequest, res: Response) => {
  try {
    // Validate request body
    const validationResult = reportSchema.safeParse(req.body);
    
    if (!validationResult.success) {
      logger.warn('Invalid report request', {
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
    
    const { url, email, domain, reason, description, threatType, severity, confidence } = validationResult.data;
    
    // Determine IOC type and value
    let iocType: string;
    let iocValue: string;
    
    if (url) {
      iocType = 'url';
      iocValue = url;
    } else if (email) {
      iocType = 'email';
      iocValue = email;
    } else if (domain) {
      iocType = 'domain';
      iocValue = domain;
    } else {
      return res.status(400).json({
        error: {
          message: 'At least one of url, email, or domain is required',
          statusCode: 400
        }
      });
    }
    
    // Report to threat intelligence service
    try {
      const reportData = {
        iocType,
        iocValue,
        threatType: threatType || 'phishing',
        severity: severity || 'medium',
        confidence: confidence || 70,
        metadata: {
          reason,
          description: description || '',
          source: 'extension',
          extensionId: req.extensionId || 'unknown',
          reportedAt: new Date().toISOString()
        }
      };
      
      logger.info('Reporting threat to threat intelligence', {
        iocType,
        iocValue: iocValue.substring(0, 100),
        extensionId: req.extensionId
      });
      
      const response = await axios.post(
        `${config.threatIntel.url}/api/v1/ioc/report`,
        reportData,
        {
          timeout: 10000,
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      
      logger.info('Threat reported successfully', {
        iocType,
        iocValue: iocValue.substring(0, 100),
        iocId: response.data?.id
      });
      
      res.json({
        success: true,
        message: 'Report submitted successfully',
        ioc: response.data
      });
    } catch (error: any) {
      logger.error('Failed to report to threat intelligence', {
        error: error.message,
        status: error.response?.status,
        iocType,
        iocValue: iocValue.substring(0, 100)
      });
      
      // Still return success to extension (reporting failure shouldn't block user)
      res.json({
        success: true,
        message: 'Report submitted (may be queued for processing)',
        warning: 'Threat intelligence service may be temporarily unavailable'
      });
    }
  } catch (error: any) {
    logger.error('Report route error', {
      error: error.message,
      stack: error.stack,
      extensionId: req.extensionId
    });
    
    res.status(500).json({
      error: {
        message: 'Report failed',
        statusCode: 500
      }
    });
  }
});

export default router;
