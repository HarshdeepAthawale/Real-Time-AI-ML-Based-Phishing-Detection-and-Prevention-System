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

export interface OAuthConfig {
  provider: 'gmail' | 'outlook';
  clientId: string;
  clientSecret: string;
  redirectUri: string;
  scopes: string[];
}

export interface OAuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
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

  // =====================================================
  // Gmail OAuth Integration
  // =====================================================

  /**
   * Generate Gmail OAuth authorization URL
   */
  getGmailAuthUrl(config: OAuthConfig): string {
    const params = new URLSearchParams({
      client_id: config.clientId,
      redirect_uri: config.redirectUri,
      response_type: 'code',
      scope: (config.scopes || [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.labels',
      ]).join(' '),
      access_type: 'offline',
      prompt: 'consent',
    });

    return `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
  }

  /**
   * Exchange Gmail authorization code for tokens
   */
  async exchangeGmailCode(
    code: string,
    config: OAuthConfig
  ): Promise<OAuthTokens> {
    const response = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        code,
        client_id: config.clientId,
        client_secret: config.clientSecret,
        redirect_uri: config.redirectUri,
        grant_type: 'authorization_code',
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Gmail OAuth token exchange failed: ${error}`);
    }

    const data = await response.json();
    return {
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt: Date.now() + data.expires_in * 1000,
    };
  }

  /**
   * Fetch Gmail messages and scan for phishing
   */
  async scanGmailInbox(
    tokens: OAuthTokens,
    scanner: EmailScannerService,
    maxResults: number = 10
  ): Promise<any[]> {
    const results: any[] = [];

    try {
      // List unread messages
      const listResponse = await fetch(
        `https://gmail.googleapis.com/gmail/v1/users/me/messages?q=is:unread&maxResults=${maxResults}`,
        { headers: { Authorization: `Bearer ${tokens.accessToken}` } }
      );

      if (!listResponse.ok) {
        throw new Error(`Gmail API error: ${listResponse.status}`);
      }

      const listData = await listResponse.json();
      const messages = listData.messages || [];

      for (const msg of messages) {
        try {
          const msgResponse = await fetch(
            `https://gmail.googleapis.com/gmail/v1/users/me/messages/${msg.id}?format=full`,
            { headers: { Authorization: `Bearer ${tokens.accessToken}` } }
          );

          if (!msgResponse.ok) continue;
          const msgData = await msgResponse.json();

          // Extract headers
          const headers = msgData.payload?.headers || [];
          const subject = headers.find((h: any) => h.name === 'Subject')?.value || '';
          const from = headers.find((h: any) => h.name === 'From')?.value || '';

          // Extract body
          let body = '';
          if (msgData.payload?.body?.data) {
            body = Buffer.from(msgData.payload.body.data, 'base64').toString('utf-8');
          } else if (msgData.payload?.parts) {
            const textPart = msgData.payload.parts.find(
              (p: any) => p.mimeType === 'text/plain'
            );
            if (textPart?.body?.data) {
              body = Buffer.from(textPart.body.data, 'base64').toString('utf-8');
            }
          }

          const emailContent = `Subject: ${subject}\nFrom: ${from}\n\n${body}`;
          const scanResult = await scanner.scanEmail(emailContent, {
            privacyMode: false,
            scanLinks: true,
            includeFullAnalysis: false,
          });

          results.push({
            messageId: msg.id,
            subject,
            from,
            isThreat: scanResult.isThreat,
            severity: scanResult.severity,
            confidence: scanResult.confidence,
          });

          if (scanResult.isThreat) {
            this.emit('threatDetected', {
              email: { subject, from, body },
              threat: scanResult,
              accountId: 'gmail',
            });
          }
        } catch (err: any) {
          logger.error('Failed to scan Gmail message', { messageId: msg.id, error: err.message });
        }
      }
    } catch (error: any) {
      logger.error('Gmail inbox scan failed', { error: error.message });
      throw error;
    }

    return results;
  }

  // =====================================================
  // Microsoft Outlook OAuth Integration
  // =====================================================

  /**
   * Generate Microsoft OAuth authorization URL
   */
  getOutlookAuthUrl(config: OAuthConfig): string {
    const params = new URLSearchParams({
      client_id: config.clientId,
      redirect_uri: config.redirectUri,
      response_type: 'code',
      scope: (config.scopes || [
        'https://graph.microsoft.com/Mail.Read',
        'offline_access',
      ]).join(' '),
      response_mode: 'query',
    });

    return `https://login.microsoftonline.com/common/oauth2/v2.0/authorize?${params.toString()}`;
  }

  /**
   * Exchange Microsoft authorization code for tokens
   */
  async exchangeOutlookCode(
    code: string,
    config: OAuthConfig
  ): Promise<OAuthTokens> {
    const response = await fetch(
      'https://login.microsoftonline.com/common/oauth2/v2.0/token',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          code,
          client_id: config.clientId,
          client_secret: config.clientSecret,
          redirect_uri: config.redirectUri,
          grant_type: 'authorization_code',
        }),
      }
    );

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Outlook OAuth token exchange failed: ${error}`);
    }

    const data = await response.json();
    return {
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt: Date.now() + data.expires_in * 1000,
    };
  }

  /**
   * Fetch Outlook messages and scan for phishing
   */
  async scanOutlookInbox(
    tokens: OAuthTokens,
    scanner: EmailScannerService,
    maxResults: number = 10
  ): Promise<any[]> {
    const results: any[] = [];

    try {
      const response = await fetch(
        `https://graph.microsoft.com/v1.0/me/messages?$filter=isRead eq false&$top=${maxResults}&$select=subject,from,body,bodyPreview`,
        { headers: { Authorization: `Bearer ${tokens.accessToken}` } }
      );

      if (!response.ok) {
        throw new Error(`Microsoft Graph API error: ${response.status}`);
      }

      const data = await response.json();
      const messages = data.value || [];

      for (const msg of messages) {
        try {
          const subject = msg.subject || '';
          const from = msg.from?.emailAddress?.address || '';
          const body = msg.body?.content || msg.bodyPreview || '';

          const emailContent = `Subject: ${subject}\nFrom: ${from}\n\n${body}`;
          const scanResult = await scanner.scanEmail(emailContent, {
            privacyMode: false,
            scanLinks: true,
            includeFullAnalysis: false,
          });

          results.push({
            messageId: msg.id,
            subject,
            from,
            isThreat: scanResult.isThreat,
            severity: scanResult.severity,
            confidence: scanResult.confidence,
          });

          if (scanResult.isThreat) {
            this.emit('threatDetected', {
              email: { subject, from, body },
              threat: scanResult,
              accountId: 'outlook',
            });
          }
        } catch (err: any) {
          logger.error('Failed to scan Outlook message', { messageId: msg.id, error: err.message });
        }
      }
    } catch (error: any) {
      logger.error('Outlook inbox scan failed', { error: error.message });
      throw error;
    }

    return results;
  }
}
