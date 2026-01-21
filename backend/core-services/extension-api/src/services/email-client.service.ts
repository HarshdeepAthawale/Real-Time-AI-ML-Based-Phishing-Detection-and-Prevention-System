import Imap from 'imap';
import { simpleParser, ParsedMail } from 'mailparser';
import { EventEmitter } from 'events';
import { EmailScannerService } from './email-scanner.service';
import { logger } from '../utils/logger';

export interface EmailAccountConfig {
  id: string;
  host: string;
  port: number;
  user: string;
  password: string;
  tls: boolean;
  tlsOptions?: {
    rejectUnauthorized?: boolean;
  };
}

export interface EmailThreatEvent {
  email: ParsedMail;
  threat: {
    isThreat: boolean;
    threatType?: string;
    severity?: string;
    confidence?: number;
    suspiciousLinks?: string[];
  };
  accountId: string;
}

export class EmailClientService extends EventEmitter {
  private connections: Map<string, Imap> = new Map();
  private scanners: Map<string, EmailScannerService> = new Map();
  private scanIntervals: Map<string, NodeJS.Timeout> = new Map();
  private isScanning: Map<string, boolean> = new Map();
  
  constructor() {
    super();
  }
  
  /**
   * Connect to an email account via IMAP
   */
  async connect(config: EmailAccountConfig): Promise<void> {
    try {
      // Close existing connection if any
      if (this.connections.has(config.id)) {
        await this.disconnect(config.id);
      }
      
      logger.info('Connecting to email account', {
        accountId: config.id,
        host: config.host,
        user: config.user
      });
      
      const imap = new Imap({
        user: config.user,
        password: config.password,
        host: config.host,
        port: config.port,
        tls: config.tls,
        tlsOptions: config.tlsOptions || { rejectUnauthorized: true },
        connTimeout: 10000,
        authTimeout: 5000
      });
      
      // Setup event handlers
      imap.once('ready', () => {
        logger.info('IMAP connection ready', { accountId: config.id });
        this.connections.set(config.id, imap);
        this.openInbox(config.id, imap);
      });
      
      imap.once('error', (err: Error) => {
        logger.error('IMAP connection error', {
          accountId: config.id,
          error: err.message
        });
        this.emit('error', { accountId: config.id, error: err });
        this.connections.delete(config.id);
      });
      
      imap.once('end', () => {
        logger.info('IMAP connection ended', { accountId: config.id });
        this.connections.delete(config.id);
        this.emit('disconnected', { accountId: config.id });
      });
      
      // Connect
      imap.connect();
    } catch (error: any) {
      logger.error('Failed to connect to email account', {
        accountId: config.id,
        error: error.message
      });
      throw error;
    }
  }
  
  /**
   * Set email scanner for an account
   */
  setScanner(accountId: string, scanner: EmailScannerService): void {
    this.scanners.set(accountId, scanner);
  }
  
  /**
   * Open inbox and start monitoring
   */
  private openInbox(accountId: string, imap: Imap): void {
    imap.openBox('INBOX', false, (err, box) => {
      if (err) {
        logger.error('Failed to open inbox', {
          accountId,
          error: err.message
        });
        this.emit('error', { accountId, error: err });
        return;
      }
      
      logger.info('Inbox opened', {
        accountId,
        totalMessages: box.messages.total
      });
      
      // Watch for new emails
      imap.on('mail', () => {
        logger.debug('New mail detected', { accountId });
        this.scanNewEmails(accountId);
      });
      
      // Initial scan
      this.scanNewEmails(accountId);
      
      // Setup periodic scanning (every 5 minutes)
      const interval = setInterval(() => {
        this.scanNewEmails(accountId);
      }, 5 * 60 * 1000);
      
      this.scanIntervals.set(accountId, interval);
    });
  }
  
  /**
   * Scan new/unseen emails
   */
  async scanNewEmails(accountId: string): Promise<void> {
    if (this.isScanning.get(accountId)) {
      logger.debug('Scan already in progress', { accountId });
      return;
    }
    
    const imap = this.connections.get(accountId);
    if (!imap) {
      logger.warn('No connection for account', { accountId });
      return;
    }
    
    const scanner = this.scanners.get(accountId);
    if (!scanner) {
      logger.warn('No scanner configured for account', { accountId });
      return;
    }
    
    this.isScanning.set(accountId, true);
    
    try {
      imap.search(['UNSEEN'], async (err, results) => {
        if (err) {
          logger.error('Failed to search emails', {
            accountId,
            error: err.message
          });
          this.isScanning.set(accountId, false);
          return;
        }
        
        if (results.length === 0) {
          logger.debug('No unseen emails', { accountId });
          this.isScanning.set(accountId, false);
          return;
        }
        
        logger.info('Scanning emails', {
          accountId,
          count: results.length
        });
        
        const fetch = imap.fetch(results, { bodies: '' });
        
        fetch.on('message', (msg) => {
          msg.on('body', async (stream: NodeJS.ReadableStream) => {
            try {
              const parsed = await simpleParser(stream as any);
              
              // Get email content
              const emailContent = parsed.text || parsed.html || parsed.textAsHtml || '';
              
              // Scan email
              const scanResult = await scanner.scanEmail(emailContent, {
                privacyMode: false,
                scanLinks: true,
                includeFullAnalysis: false
              });
              
              if (scanResult.isThreat) {
                logger.warn('Threat detected in email', {
                  accountId,
                  subject: parsed.subject,
                  from: parsed.from?.text,
                  confidence: scanResult.confidence
                });
                
                this.emit('threatDetected', {
                  email: parsed,
                  threat: scanResult,
                  accountId
                } as EmailThreatEvent);
              }
            } catch (error: any) {
              logger.error('Failed to parse/scan email', {
                accountId,
                error: error.message
              });
            }
          });
        });
        
        fetch.once('end', () => {
          logger.debug('Email scan completed', { accountId });
          this.isScanning.set(accountId, false);
        });
      });
    } catch (error: any) {
      logger.error('Email scan error', {
        accountId,
        error: error.message
      });
      this.isScanning.set(accountId, false);
    }
  }
  
  /**
   * Disconnect from an email account
   */
  async disconnect(accountId: string): Promise<void> {
    try {
      // Clear scan interval
      const interval = this.scanIntervals.get(accountId);
      if (interval) {
        clearInterval(interval);
        this.scanIntervals.delete(accountId);
      }
      
      // Close IMAP connection
      const imap = this.connections.get(accountId);
      if (imap) {
        imap.end();
        this.connections.delete(accountId);
      }
      
      // Clean up
      this.scanners.delete(accountId);
      this.isScanning.delete(accountId);
      
      logger.info('Disconnected from email account', { accountId });
    } catch (error: any) {
      logger.error('Error disconnecting email account', {
        accountId,
        error: error.message
      });
      throw error;
    }
  }
  
  /**
   * Get connection status for an account
   */
  getStatus(accountId: string): {
    connected: boolean;
    scanning: boolean;
  } {
    return {
      connected: this.connections.has(accountId),
      scanning: this.isScanning.get(accountId) || false
    };
  }
  
  /**
   * Get all connected account IDs
   */
  getConnectedAccounts(): string[] {
    return Array.from(this.connections.keys());
  }
  
  /**
   * Disconnect all accounts
   */
  async disconnectAll(): Promise<void> {
    const accountIds = Array.from(this.connections.keys());
    await Promise.all(accountIds.map(id => this.disconnect(id)));
  }
}
