import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import { logger } from './utils/logger';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3002;

// External threat intelligence feeds (placeholder URLs)
const THREAT_FEEDS = {
  // MISP feed would go here
  // OTX feed would go here
};

app.use(helmet());
app.use(cors());
app.use(express.json());

interface ThreatCheckRequest {
  url?: string;
  domain?: string;
  ip?: string;
  hash?: string;
}

interface ThreatInfo {
  is_malicious: boolean;
  threat_type?: string;
  confidence: number;
  sources: string[];
  last_seen?: string;
  description?: string;
}

// In-memory threat database (would be replaced with actual database)
const threatDatabase: Map<string, ThreatInfo> = new Map();

// Initialize with some example threats
threatDatabase.set('example-malicious.com', {
  is_malicious: true,
  threat_type: 'phishing',
  confidence: 0.9,
  sources: ['internal'],
  last_seen: new Date().toISOString(),
  description: 'Known phishing domain'
});

async function checkThreatFeeds(identifier: string, type: 'url' | 'domain' | 'ip' | 'hash'): Promise<ThreatInfo | null> {
  // TODO: Implement actual threat feed integration
  // This would query MISP, OTX, VirusTotal, etc.
  
  // Check internal database
  const cached = threatDatabase.get(identifier);
  if (cached) {
    return cached;
  }
  
  // Simulate external feed check
  logger.info(`Checking threat feeds for ${type}: ${identifier}`);
  
  // Return null if not found (would query actual feeds)
  return null;
}

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'threat-intel' });
});

app.get('/api/v1/intelligence/feeds', async (req, res) => {
  try {
    res.json({
      feeds: Object.keys(THREAT_FEEDS),
      status: 'active',
      last_updated: new Date().toISOString()
    });
  } catch (error: any) {
    logger.error(`Error fetching feeds: ${error.message}`);
    res.status(500).json({ error: 'Failed to fetch feeds' });
  }
});

app.post('/api/v1/intelligence/check', async (req, res) => {
  try {
    const request: ThreatCheckRequest = req.body;
    const results: any = {};
    
    if (request.url) {
      const parsed = new URL(request.url);
      const domainResult = await checkThreatFeeds(parsed.hostname, 'domain');
      if (domainResult) {
        results.domain = domainResult;
      }
    }
    
    if (request.domain) {
      const domainResult = await checkThreatFeeds(request.domain, 'domain');
      if (domainResult) {
        results.domain = domainResult;
      }
    }
    
    if (request.ip) {
      const ipResult = await checkThreatFeeds(request.ip, 'ip');
      if (ipResult) {
        results.ip = ipResult;
      }
    }
    
    if (request.hash) {
      const hashResult = await checkThreatFeeds(request.hash, 'hash');
      if (hashResult) {
        results.hash = hashResult;
      }
    }
    
    // If no threats found, return safe result
    if (Object.keys(results).length === 0) {
      res.json({
        is_malicious: false,
        confidence: 0.1,
        sources: [],
        results: {}
      });
    } else {
      // Aggregate results
      const threats = Object.values(results) as ThreatInfo[];
      const isMalicious = threats.some(t => t.is_malicious);
      const maxConfidence = Math.max(...threats.map(t => t.confidence));
      
      res.json({
        is_malicious: isMalicious,
        confidence: maxConfidence,
        sources: threats.flatMap(t => t.sources),
        results
      });
    }
  } catch (error: any) {
    logger.error(`Threat check error: ${error.message}`);
    res.status(500).json({ error: 'Threat check failed', message: error.message });
  }
});

app.post('/api/v1/intelligence/report', async (req, res) => {
  try {
    const { identifier, threat_type, confidence, description } = req.body;
    
    // Add to internal database
    threatDatabase.set(identifier, {
      is_malicious: true,
      threat_type: threat_type || 'unknown',
      confidence: confidence || 0.8,
      sources: ['user-report'],
      last_seen: new Date().toISOString(),
      description
    });
    
    logger.info(`Threat reported: ${identifier}`);
    res.json({ success: true, message: 'Threat reported successfully' });
  } catch (error: any) {
    logger.error(`Report error: ${error.message}`);
    res.status(500).json({ error: 'Report failed', message: error.message });
  }
});

app.listen(PORT, () => {
  logger.info(`Threat Intelligence service running on port ${PORT}`);
});
