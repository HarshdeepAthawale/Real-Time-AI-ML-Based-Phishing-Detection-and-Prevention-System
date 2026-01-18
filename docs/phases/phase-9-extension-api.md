# Phase 9: Browser Extension Backend & Edge Integration

## Objective
Build lightweight backend services for browser extension and email client integration, providing real-time URL checking, email scanning, and privacy-preserving analysis.

## Prerequisites
- Phase 6 completed (Detection API)
- Browser extension development knowledge
- Email client API access (IMAP/POP3)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         Extension API Service                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   URL        │  │   Email       │  │   Reporting  │  │
│  │   Checker    │  │   Scanner    │  │   Handler    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Local      │  │   Privacy    │  │   Config     │  │
│  │   Cache      │  │   Filter     │  │   Manager    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
backend/core-services/extension-api/
├── src/
│   ├── index.ts
│   ├── routes/
│   │   ├── url-check.routes.ts
│   │   ├── email-scan.routes.ts
│   │   └── report.routes.ts
│   ├── services/
│   │   ├── url-checker.service.ts
│   │   ├── email-scanner.service.ts
│   │   └── privacy-filter.service.ts
│   └── middleware/
│       └── extension-auth.middleware.ts
│
extensions/
├── chrome/
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   └── popup.html
├── firefox/
│   └── (similar structure)
└── edge/
    └── (similar structure)
```

## Implementation Steps

### 1. Extension API Service

**File**: `backend/core-services/extension-api/src/index.ts`

```typescript
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { config } from './config';
import urlCheckRoutes from './routes/url-check.routes';
import emailScanRoutes from './routes/email-scan.routes';
import reportRoutes from './routes/report.routes';
import { extensionAuthMiddleware } from './middleware/extension-auth.middleware';

const app = express();

app.use(helmet());
app.use(cors({
  origin: [
    'chrome-extension://*',
    'moz-extension://*',
    'ms-browser-extension://*'
  ],
  credentials: true
}));
app.use(express.json({ limit: '1mb' }));

// Routes
app.use('/api/v1/extension/check-url', extensionAuthMiddleware, urlCheckRoutes);
app.use('/api/v1/extension/scan-email', extensionAuthMiddleware, emailScanRoutes);
app.use('/api/v1/extension/report', extensionAuthMiddleware, reportRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

const PORT = config.port || 3001;
app.listen(PORT, () => {
  console.log(`Extension API running on port ${PORT}`);
});
```

### 2. URL Checker Service

**File**: `backend/core-services/extension-api/src/services/url-checker.service.ts`

```typescript
import axios from 'axios';
import { CacheService } from './cache.service';
import { PrivacyFilterService } from './privacy-filter.service';
import { logger } from '../utils/logger';

export class URLCheckerService {
  private detectionApiUrl: string;
  private cache: CacheService;
  private privacyFilter: PrivacyFilterService;
  
  constructor(detectionApiUrl: string, cache: CacheService, privacyFilter: PrivacyFilterService) {
    this.detectionApiUrl = detectionApiUrl;
    this.cache = cache;
    this.privacyFilter = privacyFilter;
  }
  
  async checkURL(url: string, options: {
    includeFullAnalysis?: boolean;
    privacyMode?: boolean;
  } = {}): Promise<{
    isThreat: boolean;
    severity?: string;
    confidence?: number;
    cached: boolean;
  }> {
    // Check cache first
    const cacheKey = `url:${this.hashURL(url)}`;
    const cached = await this.cache.get(cacheKey);
    
    if (cached) {
      return {
        ...cached,
        cached: true
      };
    }
    
    // Privacy filter - only send minimal data if privacy mode
    const urlToCheck = options.privacyMode 
      ? this.privacyFilter.filterURL(url)
      : url;
    
    try {
      // Call detection API
      const response = await axios.post(`${this.detectionApiUrl}/api/v1/detect/url`, {
        url: urlToCheck,
        includeFullAnalysis: options.includeFullAnalysis || false
      }, {
        timeout: 5000 // Fast timeout for extension
      });
      
      const result = {
        isThreat: response.data.isThreat,
        severity: response.data.severity,
        confidence: response.data.confidence,
        cached: false
      };
      
      // Cache result (shorter TTL for extension API)
      await this.cache.set(cacheKey, result, 1800); // 30 minutes
      
      return result;
    } catch (error) {
      logger.error('URL check failed', error);
      // Return safe default
      return {
        isThreat: false,
        cached: false
      };
    }
  }
  
  private hashURL(url: string): string {
    // Simple hash for cache key
    const crypto = require('crypto');
    return crypto.createHash('sha256').update(url).digest('hex').substring(0, 16);
  }
}
```

### 3. Privacy Filter Service

**File**: `backend/core-services/extension-api/src/services/privacy-filter.service.ts`

```typescript
import { URL } from 'url';

export class PrivacyFilterService {
  filterURL(url: string): string {
    // Extract only domain, remove path and query params
    try {
      const parsed = new URL(url);
      return `${parsed.protocol}//${parsed.hostname}`;
    } catch {
      return url;
    }
  }
  
  filterEmailContent(emailContent: string): {
    subject: string;
    from: string;
    hasLinks: boolean;
    linkCount: number;
  } {
    // Extract minimal information
    const subjectMatch = emailContent.match(/Subject:\s*(.+)/i);
    const fromMatch = emailContent.match(/From:\s*(.+)/i);
    const linkPattern = /https?:\/\/[^\s<>"{}|\\^`\[\]]+/g;
    const links = emailContent.match(linkPattern) || [];
    
    return {
      subject: subjectMatch ? subjectMatch[1] : '',
      from: fromMatch ? fromMatch[1] : '',
      hasLinks: links.length > 0,
      linkCount: links.length
    };
  }
}
```

### 4. Email Scanner Service

**File**: `backend/core-services/extension-api/src/services/email-scanner.service.ts`

```typescript
import axios from 'axios';
import { PrivacyFilterService } from './privacy-filter.service';
import { logger } from '../utils/logger';

export class EmailScannerService {
  private detectionApiUrl: string;
  private privacyFilter: PrivacyFilterService;
  
  constructor(detectionApiUrl: string, privacyFilter: PrivacyFilterService) {
    this.detectionApiUrl = detectionApiUrl;
    this.privacyFilter = privacyFilter;
  }
  
  async scanEmail(
    emailContent: string,
    options: {
      privacyMode?: boolean;
      scanLinks?: boolean;
    } = {}
  ): Promise<{
    isThreat: boolean;
    threatType?: string;
    severity?: string;
    suspiciousLinks?: string[];
  }> {
    try {
      // Filter email content if privacy mode
      const contentToScan = options.privacyMode
        ? this.privacyFilter.filterEmailContent(emailContent).subject
        : emailContent;
      
      // Call detection API
      const response = await axios.post(`${this.detectionApiUrl}/api/v1/detect/email`, {
        emailContent: contentToScan,
        includeFullAnalysis: false
      }, {
        timeout: 10000
      });
      
      // Extract suspicious links if requested
      let suspiciousLinks: string[] = [];
      if (options.scanLinks && response.data.indicators) {
        const linkPattern = /https?:\/\/[^\s<>"{}|\\^`\[\]]+/g;
        const links = emailContent.match(linkPattern) || [];
        
        // Check each link
        for (const link of links) {
          const linkCheck = await this.checkLink(link);
          if (linkCheck.isThreat) {
            suspiciousLinks.push(link);
          }
        }
      }
      
      return {
        isThreat: response.data.isThreat,
        threatType: response.data.threatType,
        severity: response.data.severity,
        suspiciousLinks: suspiciousLinks.length > 0 ? suspiciousLinks : undefined
      };
    } catch (error) {
      logger.error('Email scan failed', error);
      return {
        isThreat: false
      };
    }
  }
  
  private async checkLink(url: string): Promise<{ isThreat: boolean }> {
    // Quick link check
    // Implementation similar to URL checker
    return { isThreat: false };
  }
}
```

### 5. Browser Extension (Chrome)

**File**: `extensions/chrome/manifest.json`

```json
{
  "manifest_version": 3,
  "name": "Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System",
  "version": "1.0.0",
  "description": "Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System",
  "permissions": [
    "tabs",
    "activeTab",
    "storage",
    "webRequest",
    "alarms"
  ],
  "host_permissions": [
    "http://*/*",
    "https://*/*"
  ],
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [{
    "matches": ["<all_urls>"],
    "js": ["content.js"]
  }],
  "action": {
    "default_popup": "popup.html",
    "default_icon": {
      "16": "icons/icon16.png",
      "48": "icons/icon48.png",
      "128": "icons/icon128.png"
    }
  },
  "options_page": "options.html"
}
```

**File**: `extensions/chrome/background.js`

```javascript
const API_URL = 'https://api.yourdomain.com/api/v1/extension';

// Check URL when tab is updated
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    await checkURL(tab.url, tabId);
  }
});

// Check URL
async function checkURL(url, tabId) {
  try {
    const response = await fetch(`${API_URL}/check-url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Extension-Id': chrome.runtime.id
      },
      body: JSON.stringify({ url })
    });
    
    const result = await response.json();
    
    if (result.isThreat) {
      // Show warning badge
      chrome.action.setBadgeText({
        text: '!',
        tabId: tabId
      });
      chrome.action.setBadgeBackgroundColor({
        color: '#ff0000',
        tabId: tabId
      });
      
      // Store threat info
      chrome.storage.local.set({
        [`threat_${tabId}`]: result
      });
    } else {
      chrome.action.setBadgeText({
        text: '',
        tabId: tabId
      });
    }
  } catch (error) {
    console.error('URL check failed', error);
  }
}

// Intercept navigation to suspicious URLs
chrome.webRequest.onBeforeRequest.addListener(
  async (details) => {
    const result = await checkURL(details.url, details.tabId);
    if (result.isThreat && result.severity === 'critical') {
      return { cancel: true }; // Block navigation
    }
  },
  { urls: ["<all_urls>"] },
  ["blocking"]
);
```

**File**: `extensions/chrome/content.js`

```javascript
// Content script to detect phishing forms
(function() {
  // Check for suspicious forms
  const forms = document.querySelectorAll('form');
  forms.forEach(form => {
    const passwordFields = form.querySelectorAll('input[type="password"]');
    if (passwordFields.length > 0) {
      // Check if form is suspicious
      checkForm(form);
    }
  });
  
  async function checkForm(form) {
    const formData = {
      action: form.action,
      method: form.method,
      fields: Array.from(form.querySelectorAll('input')).map(input => ({
        type: input.type,
        name: input.name
      }))
    };
    
    // Send to extension API for analysis
    chrome.runtime.sendMessage({
      type: 'checkForm',
      data: formData
    }, (response) => {
      if (response.isSuspicious) {
        showWarning(form);
      }
    });
  }
  
  function showWarning(form) {
    const warning = document.createElement('div');
    warning.className = 'phishing-warning';
    warning.innerHTML = `
      <div style="background: #ff0000; color: white; padding: 10px; margin: 10px 0;">
        ⚠️ Warning: This form may be a phishing attempt. Proceed with caution.
      </div>
    `;
    form.parentNode.insertBefore(warning, form);
  }
})();
```

### 6. Email Client Integration

**File**: `backend/core-services/extension-api/src/services/email-client.service.ts`

```typescript
import Imap from 'imap';
import { simpleParser } from 'mailparser';
import { EventEmitter } from 'events';
import { EmailScannerService } from './email-scanner.service';

export class EmailClientService extends EventEmitter {
  private imap: Imap;
  private scanner: EmailScannerService;
  private isConnected: boolean = false;
  
  constructor(config: {
    host: string;
    port: number;
    user: string;
    password: string;
    tls: boolean;
  }, scanner: EmailScannerService) {
    super();
    this.scanner = scanner;
    
    this.imap = new Imap({
      user: config.user,
      password: config.password,
      host: config.host,
      port: config.port,
      tls: config.tls
    });
    
    this.setupEventHandlers();
  }
  
  private setupEventHandlers(): void {
    this.imap.once('ready', () => {
      this.isConnected = true;
      this.emit('connected');
      this.openInbox();
    });
    
    this.imap.once('error', (err) => {
      this.emit('error', err);
    });
    
    this.imap.once('end', () => {
      this.isConnected = false;
      this.emit('disconnected');
    });
  }
  
  connect(): void {
    this.imap.connect();
  }
  
  private openInbox(): void {
    this.imap.openBox('INBOX', false, (err, box) => {
      if (err) {
        this.emit('error', err);
        return;
      }
      
      // Watch for new emails
      this.imap.on('mail', () => {
        this.scanNewEmails();
      });
      
      // Initial scan
      this.scanNewEmails();
    });
  }
  
  private async scanNewEmails(): Promise<void> {
    this.imap.search(['UNSEEN'], async (err, results) => {
      if (err) {
        this.emit('error', err);
        return;
      }
      
      if (results.length === 0) return;
      
      const fetch = this.imap.fetch(results, { bodies: '' });
      
      fetch.on('message', (msg) => {
        msg.on('body', async (stream) => {
          const parsed = await simpleParser(stream);
          
          // Scan email
          const scanResult = await this.scanner.scanEmail(parsed.text || parsed.html || '');
          
          if (scanResult.isThreat) {
            this.emit('threatDetected', {
              email: parsed,
              threat: scanResult
            });
          }
        });
      });
    });
  }
}
```

## Deliverables Checklist

- [ ] Extension API service created
- [ ] URL checker service implemented
- [ ] Email scanner service implemented
- [ ] Privacy filter implemented
- [ ] Chrome extension created
- [ ] Firefox extension created
- [ ] Edge extension created
- [ ] Email client integration
- [ ] Local caching for offline support
- [ ] Tests written

## Next Steps

After completing Phase 9:
1. Publish browser extensions to stores
2. Test email client integrations
3. Monitor extension usage
4. Proceed to Phase 10: Sandbox Integration
