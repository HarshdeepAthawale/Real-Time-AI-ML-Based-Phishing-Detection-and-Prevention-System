import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { EmailClientService, EmailAccountConfig } from '../services/email-client.service';
import { EmailScannerService } from '../services/email-scanner.service';
import { ExtensionRequest } from '../middleware/extension-auth.middleware';
import { logger } from '../utils/logger';

const router = Router();

// Validation schema for email connection
const emailConnectSchema = z.object({
  id: z.string().min(1, 'Account ID is required'),
  host: z.string().min(1, 'Host is required'),
  port: z.number().int().min(1).max(65535),
  user: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
  tls: z.boolean().optional().default(true),
  tlsOptions: z.object({
    rejectUnauthorized: z.boolean().optional()
  }).optional()
});

/**
 * POST /api/v1/extension/email/connect
 * Connect to an email account via IMAP
 */
router.post('/connect', async (req: ExtensionRequest, res: Response) => {
  try {
    const validationResult = emailConnectSchema.safeParse(req.body);
    
    if (!validationResult.success) {
      return res.status(400).json({
        error: {
          message: 'Validation failed',
          statusCode: 400,
          details: validationResult.error.errors
        }
      });
    }
    
    const config: EmailAccountConfig = validationResult.data;
    
    // Get services from app context
    const emailClient = req.app.get('emailClient') as EmailClientService;
    const emailScanner = req.app.get('emailScanner') as EmailScannerService;
    
    if (!emailClient || !emailScanner) {
      logger.error('Email client or scanner service not available');
      return res.status(500).json({
        error: {
          message: 'Service unavailable',
          statusCode: 500
        }
      });
    }
    
    // Set scanner for this account
    emailClient.setScanner(config.id, emailScanner);
    
    // Setup threat detection listener
    emailClient.on('threatDetected', (event) => {
      logger.warn('Email threat detected', {
        accountId: event.accountId,
        subject: event.email.subject,
        threatType: event.threat.threatType
      });
    });
    
    // Connect
    await emailClient.connect(config);
    
    logger.info('Email account connected', {
      accountId: config.id,
      extensionId: req.extensionId
    });
    
    res.json({
      success: true,
      message: 'Email account connected successfully',
      accountId: config.id
    });
  } catch (error: any) {
    logger.error('Email connection error', {
      error: error.message,
      extensionId: req.extensionId
    });
    
    res.status(500).json({
      error: {
        message: 'Failed to connect email account',
        statusCode: 500,
        details: error.message
      }
    });
  }
});

/**
 * POST /api/v1/extension/email/disconnect
 * Disconnect from an email account
 */
router.post('/disconnect', async (req: ExtensionRequest, res: Response) => {
  try {
    const { accountId } = req.body;
    
    if (!accountId) {
      return res.status(400).json({
        error: {
          message: 'Account ID is required',
          statusCode: 400
        }
      });
    }
    
    const emailClient = req.app.get('emailClient') as EmailClientService;
    
    if (!emailClient) {
      return res.status(500).json({
        error: {
          message: 'Service unavailable',
          statusCode: 500
        }
      });
    }
    
    await emailClient.disconnect(accountId);
    
    res.json({
      success: true,
      message: 'Email account disconnected successfully'
    });
  } catch (error: any) {
    logger.error('Email disconnection error', {
      error: error.message,
      extensionId: req.extensionId
    });
    
    res.status(500).json({
      error: {
        message: 'Failed to disconnect email account',
        statusCode: 500
      }
    });
  }
});

/**
 * GET /api/v1/extension/email/status
 * Get connection status for email accounts
 */
router.get('/status', async (req: ExtensionRequest, res: Response) => {
  try {
    const { accountId } = req.query;
    
    const emailClient = req.app.get('emailClient') as EmailClientService;
    
    if (!emailClient) {
      return res.status(500).json({
        error: {
          message: 'Service unavailable',
          statusCode: 500
        }
      });
    }
    
    if (accountId) {
      // Get status for specific account
      const status = emailClient.getStatus(accountId as string);
      res.json({
        accountId,
        ...status
      });
    } else {
      // Get all connected accounts
      const accounts = emailClient.getConnectedAccounts();
      const statuses = accounts.map(id => ({
        accountId: id,
        ...emailClient.getStatus(id)
      }));
      
      res.json({
        accounts: statuses,
        total: accounts.length
      });
    }
  } catch (error: any) {
    logger.error('Email status error', {
      error: error.message,
      extensionId: req.extensionId
    });
    
    res.status(500).json({
      error: {
        message: 'Failed to get email status',
        statusCode: 500
      }
    });
  }
});

/**
 * POST /api/v1/extension/email/scan
 * Manually trigger email scan for an account
 */
router.post('/scan', async (req: ExtensionRequest, res: Response) => {
  try {
    const { accountId } = req.body;
    
    if (!accountId) {
      return res.status(400).json({
        error: {
          message: 'Account ID is required',
          statusCode: 400
        }
      });
    }
    
    const emailClient = req.app.get('emailClient') as EmailClientService;
    
    if (!emailClient) {
      return res.status(500).json({
        error: {
          message: 'Service unavailable',
          statusCode: 500
        }
      });
    }
    
    // Trigger scan
    await emailClient.scanNewEmails(accountId);
    
    res.json({
      success: true,
      message: 'Email scan triggered successfully'
    });
  } catch (error: any) {
    logger.error('Email scan trigger error', {
      error: error.message,
      extensionId: req.extensionId
    });
    
    res.status(500).json({
      error: {
        message: 'Failed to trigger email scan',
        statusCode: 500
      }
    });
  }
});

export default router;
