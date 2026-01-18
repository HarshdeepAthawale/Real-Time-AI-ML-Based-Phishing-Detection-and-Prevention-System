# Phase 7: Threat Intelligence Integration Service

## Objective
Integrate with external threat intelligence feeds (MISP, AlienVault OTX), build an IOC management system with fast lookup engine, and implement feed synchronization.

## Prerequisites
- Phase 1 & 2 completed
- Phase 6 completed (for IOC matching)
- Access to MISP and/or AlienVault OTX APIs
- Node.js 20+ and TypeScript

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│      Threat Intelligence Service                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   MISP       │  │   OTX        │  │   Custom     │  │
│  │   Connector  │  │   Connector  │  │   Feeds      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   IOC        │  │   Bloom      │  │   Sync       │  │
│  │   Manager    │  │   Filter     │  │   Scheduler  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
backend/core-services/threat-intel/
├── src/
│   ├── index.ts
│   ├── integrations/
│   │   ├── misp.client.ts          # MISP API client
│   │   ├── otx.client.ts           # AlienVault OTX client
│   │   └── base-feed.client.ts     # Base class for feeds
│   ├── services/
│   │   ├── ioc-manager.service.ts  # IOC CRUD operations
│   │   ├── ioc-matcher.service.ts # Fast IOC matching
│   │   ├── sync.service.ts         # Feed synchronization
│   │   └── enrichment.service.ts  # IOC enrichment
│   ├── models/
│   │   └── ioc.model.ts
│   ├── utils/
│   │   ├── bloom-filter.ts         # Bloom filter for fast lookups
│   │   └── normalizers.ts         # IOC normalization
│   └── jobs/
│       └── sync-scheduler.ts       # Scheduled sync jobs
├── tests/
├── Dockerfile
├── package.json
└── README.md
```

## Implementation Steps

### 1. Dependencies

**File**: `backend/core-services/threat-intel/package.json`

```json
{
  "dependencies": {
    "express": "^4.18.2",
    "axios": "^1.6.2",
    "ioredis": "^5.3.2",
    "bloom-filters": "^1.3.0",
    "node-cron": "^3.0.3",
    "pg": "^8.11.3",
    "zod": "^3.22.4",
    "winston": "^3.11.0"
  }
}
```

### 2. MISP Client

**File**: `backend/core-services/threat-intel/src/integrations/misp.client.ts`

```typescript
import axios, { AxiosInstance } from 'axios';
import { IOC } from '../models/ioc.model';
import { logger } from '../utils/logger';

export class MISPClient {
  private client: AxiosInstance;
  private apiKey: string;
  private baseURL: string;
  
  constructor(baseURL: string, apiKey: string) {
    this.baseURL = baseURL;
    this.apiKey = apiKey;
    this.client = axios.create({
      baseURL: `${baseURL}/attributes/restSearch`,
      headers: {
        'Authorization': apiKey,
        'Content-Type': 'application/json'
      }
    });
  }
  
  async fetchIOCs(since?: Date): Promise<IOC[]> {
    try {
      const params: any = {
        returnFormat: 'json',
        type: ['url', 'domain', 'ip-dst', 'ip-src', 'md5', 'sha1', 'sha256', 'filename']
      };
      
      if (since) {
        params.timestamp = Math.floor(since.getTime() / 1000);
      }
      
      const response = await this.client.post('', params);
      const attributes = response.data.response?.Attribute || [];
      
      return attributes.map((attr: any) => this.mapToIOC(attr));
    } catch (error) {
      logger.error('MISP fetch error', error);
      throw error;
    }
  }
  
  async publishIOC(ioc: IOC): Promise<void> {
    try {
      const event = {
        Event: {
          info: `Phishing detection: ${ioc.iocType}`,
          distribution: 1, // Your organization only
          threat_level_id: this.mapSeverityToThreatLevel(ioc.severity),
          Attribute: [{
            type: this.mapIOCTypeToMISP(ioc.iocType),
            value: ioc.iocValue,
            category: 'Network activity',
            to_ids: true
          }]
        }
      };
      
      await this.client.post('/events/add', event);
    } catch (error) {
      logger.error('MISP publish error', error);
      throw error;
    }
  }
  
  private mapToIOC(attr: any): IOC {
    return {
      iocType: this.mapMISPTypeToIOC(attr.type),
      iocValue: attr.value,
      threatType: attr.Event?.info || 'unknown',
      severity: this.mapThreatLevelToSeverity(attr.Event?.threat_level_id),
      source: 'misp',
      firstSeenAt: new Date(attr.timestamp * 1000),
      metadata: {
        eventId: attr.event_id,
        attributeId: attr.id
      }
    };
  }
  
  private mapIOCTypeToMISP(type: string): string {
    const mapping: Record<string, string> = {
      'url': 'url',
      'domain': 'domain',
      'ip': 'ip-dst',
      'hash_md5': 'md5',
      'hash_sha1': 'sha1',
      'hash_sha256': 'sha256',
      'filename': 'filename'
    };
    return mapping[type] || type;
  }
  
  private mapMISPTypeToIOC(type: string): string {
    const mapping: Record<string, string> = {
      'url': 'url',
      'domain': 'domain',
      'ip-dst': 'ip',
      'ip-src': 'ip',
      'md5': 'hash_md5',
      'sha1': 'hash_sha1',
      'sha256': 'hash_sha256',
      'filename': 'filename'
    };
    return mapping[type] || type;
  }
  
  private mapSeverityToThreatLevel(severity: string): number {
    const mapping: Record<string, number> = {
      'critical': 1,
      'high': 2,
      'medium': 3,
      'low': 4
    };
    return mapping[severity] || 4;
  }
  
  private mapThreatLevelToSeverity(level: number): string {
    const mapping: Record<number, string> = {
      1: 'critical',
      2: 'high',
      3: 'medium',
      4: 'low'
    };
    return mapping[level] || 'low';
  }
}
```

### 3. OTX Client

**File**: `backend/core-services/threat-intel/src/integrations/otx.client.ts`

```typescript
import axios, { AxiosInstance } from 'axios';
import { IOC } from '../models/ioc.model';
import { logger } from '../utils/logger';

export class OTXClient {
  private client: AxiosInstance;
  private apiKey: string;
  
  constructor(apiKey: string) {
    this.apiKey = apiKey;
    this.client = axios.create({
      baseURL: 'https://otx.alienvault.com/api/v1',
      headers: {
        'X-OTX-API-KEY': apiKey
      }
    });
  }
  
  async fetchPulses(since?: Date): Promise<IOC[]> {
    try {
      const params: any = {
        limit: 100
      };
      
      if (since) {
        params.modified_since = since.toISOString();
      }
      
      const response = await this.client.get('/pulses/subscribed', { params });
      const pulses = response.data.results || [];
      
      const iocs: IOC[] = [];
      for (const pulse of pulses) {
        for (const indicator of pulse.indicators || []) {
          iocs.push(this.mapToIOC(indicator, pulse));
        }
      }
      
      return iocs;
    } catch (error) {
      logger.error('OTX fetch error', error);
      throw error;
    }
  }
  
  private mapToIOC(indicator: any, pulse: any): IOC {
    return {
      iocType: indicator.type,
      iocValue: indicator.indicator,
      threatType: pulse.name,
      severity: this.mapPulseSeverity(pulse.tlp),
      source: 'otx',
      firstSeenAt: new Date(indicator.created),
      metadata: {
        pulseId: pulse.id,
        pulseName: pulse.name,
        tlp: pulse.tlp
      }
    };
  }
  
  private mapPulseSeverity(tlp: string): string {
    // Map TLP to severity
    const mapping: Record<string, string> = {
      'RED': 'critical',
      'AMBER': 'high',
      'GREEN': 'medium',
      'WHITE': 'low'
    };
    return mapping[tlp] || 'low';
  }
}
```

### 4. IOC Manager Service

**File**: `backend/core-services/threat-intel/src/services/ioc-manager.service.ts`

```typescript
import { Pool } from 'pg';
import { IOC } from '../models/ioc.model';
import { logger } from '../utils/logger';
import crypto from 'crypto';

export class IOCManagerService {
  private pool: Pool;
  
  constructor(pool: Pool) {
    this.pool = pool;
  }
  
  async createIOC(ioc: IOC): Promise<IOC> {
    const iocValueHash = this.hashIOCValue(ioc.iocValue);
    
    const query = `
      INSERT INTO iocs (
        feed_id, ioc_type, ioc_value, ioc_value_hash,
        threat_type, severity, confidence, first_seen_at,
        source_reports, metadata
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
      ON CONFLICT (ioc_type, ioc_value_hash) 
      DO UPDATE SET
        last_seen_at = CURRENT_TIMESTAMP,
        source_reports = iocs.source_reports + 1,
        metadata = jsonb_merge(iocs.metadata, $10)
      RETURNING *
    `;
    
    const values = [
      ioc.feedId || null,
      ioc.iocType,
      ioc.iocValue,
      iocValueHash,
      ioc.threatType,
      ioc.severity,
      ioc.confidence || 50,
      ioc.firstSeenAt || new Date(),
      1,
      JSON.stringify(ioc.metadata || {})
    ];
    
    const result = await this.pool.query(query, values);
    return this.mapRowToIOC(result.rows[0]);
  }
  
  async findIOC(iocType: string, iocValue: string): Promise<IOC | null> {
    const iocValueHash = this.hashIOCValue(iocValue);
    
    const query = `
      SELECT * FROM iocs
      WHERE ioc_type = $1 AND ioc_value_hash = $2
    `;
    
    const result = await this.pool.query(query, [iocType, iocValueHash]);
    
    if (result.rows.length === 0) {
      return null;
    }
    
    return this.mapRowToIOC(result.rows[0]);
  }
  
  async bulkCreateIOCs(iocs: IOC[]): Promise<number> {
    let inserted = 0;
    
    for (const ioc of iocs) {
      try {
        await this.createIOC(ioc);
        inserted++;
      } catch (error) {
        logger.error(`Failed to insert IOC: ${ioc.iocValue}`, error);
      }
    }
    
    return inserted;
  }
  
  private hashIOCValue(value: string): string {
    return crypto.createHash('sha256').update(value.toLowerCase().trim()).digest('hex');
  }
  
  private mapRowToIOC(row: any): IOC {
    return {
      id: row.id,
      feedId: row.feed_id,
      iocType: row.ioc_type,
      iocValue: row.ioc_value,
      threatType: row.threat_type,
      severity: row.severity,
      confidence: row.confidence,
      firstSeenAt: row.first_seen_at,
      lastSeenAt: row.last_seen_at,
      source: row.feed_id ? 'feed' : 'user',
      sourceReports: row.source_reports,
      metadata: row.metadata || {}
    };
  }
}
```

### 5. IOC Matcher Service (with Bloom Filter)

**File**: `backend/core-services/threat-intel/src/services/ioc-matcher.service.ts`

```typescript
import Redis from 'ioredis';
import { BloomFilter } from 'bloom-filters';
import { IOCManagerService } from './ioc-manager.service';
import { logger } from '../utils/logger';

export class IOCMatcherService {
  private redis: Redis;
  private iocManager: IOCManagerService;
  private bloomFilters: Map<string, BloomFilter> = new Map();
  
  constructor(redis: Redis, iocManager: IOCManagerService) {
    this.redis = redis;
    this.iocManager = iocManager;
    this.initializeBloomFilters();
  }
  
  private async initializeBloomFilters(): Promise<void> {
    // Initialize bloom filters for each IOC type
    const types = ['url', 'domain', 'ip', 'hash_md5', 'hash_sha1', 'hash_sha256'];
    
    for (const type of types) {
      await this.loadBloomFilter(type);
    }
  }
  
  private async loadBloomFilter(iocType: string): Promise<void> {
    try {
      // Try to load from Redis
      const serialized = await this.redis.get(`bloom:${iocType}`);
      
      if (serialized) {
        this.bloomFilters.set(iocType, BloomFilter.fromJSON(JSON.parse(serialized)));
      } else {
        // Create new bloom filter
        const filter = BloomFilter.create(1000000, 0.01); // 1M items, 1% false positive rate
        this.bloomFilters.set(iocType, filter);
      }
    } catch (error) {
      logger.error(`Failed to load bloom filter for ${iocType}`, error);
    }
  }
  
  async matchIOC(iocType: string, iocValue: string): Promise<IOC | null> {
    // Fast negative lookup using bloom filter
    const filter = this.bloomFilters.get(iocType);
    if (filter && !filter.has(iocValue.toLowerCase().trim())) {
      return null; // Definitely not in database
    }
    
    // If bloom filter says it might exist, check database
    return await this.iocManager.findIOC(iocType, iocValue);
  }
  
  async addToBloomFilter(iocType: string, iocValue: string): Promise<void> {
    const filter = this.bloomFilters.get(iocType);
    if (filter) {
      filter.add(iocValue.toLowerCase().trim());
      
      // Persist to Redis
      await this.redis.set(
        `bloom:${iocType}`,
        JSON.stringify(filter.saveAsJSON()),
        'EX',
        86400 * 7 // 7 days
      );
    }
  }
  
  async bulkAddToBloomFilter(iocType: string, iocValues: string[]): Promise<void> {
    const filter = this.bloomFilters.get(iocType);
    if (filter) {
      for (const value of iocValues) {
        filter.add(value.toLowerCase().trim());
      }
      
      // Persist to Redis
      await this.redis.set(
        `bloom:${iocType}`,
        JSON.stringify(filter.saveAsJSON()),
        'EX',
        86400 * 7
      );
    }
  }
}
```

### 6. Sync Service

**File**: `backend/core-services/threat-intel/src/services/sync.service.ts`

```typescript
import { MISPClient } from '../integrations/misp.client';
import { OTXClient } from '../integrations/otx.client';
import { IOCManagerService } from './ioc-manager.service';
import { IOCMatcherService } from './ioc-matcher.service';
import { logger } from '../utils/logger';

export class SyncService {
  private mispClient?: MISPClient;
  private otxClient?: OTXClient;
  private iocManager: IOCManagerService;
  private iocMatcher: IOCMatcherService;
  
  constructor(
    iocManager: IOCManagerService,
    iocMatcher: IOCMatcherService,
    mispConfig?: { baseURL: string; apiKey: string },
    otxConfig?: { apiKey: string }
  ) {
    this.iocManager = iocManager;
    this.iocMatcher = iocMatcher;
    
    if (mispConfig) {
      this.mispClient = new MISPClient(mispConfig.baseURL, mispConfig.apiKey);
    }
    
    if (otxConfig) {
      this.otxClient = new OTXClient(otxConfig.apiKey);
    }
  }
  
  async syncAllFeeds(): Promise<{ misp: number; otx: number }> {
    const results = { misp: 0, otx: 0 };
    
    // Sync MISP
    if (this.mispClient) {
      try {
        const lastSync = await this.getLastSyncTime('misp');
        const iocs = await this.mispClient.fetchIOCs(lastSync);
        const inserted = await this.iocManager.bulkCreateIOCs(iocs);
        results.misp = inserted;
        
        // Update bloom filters
        const iocValues = iocs.map(ioc => ioc.iocValue);
        await this.iocMatcher.bulkAddToBloomFilter('domain', iocValues.filter((_, i) => iocs[i].iocType === 'domain'));
        
        await this.updateLastSyncTime('misp');
        logger.info(`MISP sync completed: ${inserted} IOCs`);
      } catch (error) {
        logger.error('MISP sync failed', error);
      }
    }
    
    // Sync OTX
    if (this.otxClient) {
      try {
        const lastSync = await this.getLastSyncTime('otx');
        const iocs = await this.otxClient.fetchPulses(lastSync);
        const inserted = await this.iocManager.bulkCreateIOCs(iocs);
        results.otx = inserted;
        
        await this.updateLastSyncTime('otx');
        logger.info(`OTX sync completed: ${inserted} IOCs`);
      } catch (error) {
        logger.error('OTX sync failed', error);
      }
    }
    
    return results;
  }
  
  private async getLastSyncTime(feedName: string): Promise<Date | undefined> {
    // Get from database or Redis
    // Implementation depends on your storage choice
    return undefined; // Sync all if not set
  }
  
  private async updateLastSyncTime(feedName: string): Promise<void> {
    // Update last sync time
  }
}
```

### 7. API Routes

**File**: `backend/core-services/threat-intel/src/routes/ioc.routes.ts`

```typescript
import { Router, Request, Response } from 'express';
import { IOCMatcherService } from '../services/ioc-matcher.service';
import { IOCManagerService } from '../services/ioc-manager.service';
import { z } from 'zod';

const router = Router();

const checkIOCSchema = z.object({
  iocType: z.enum(['url', 'domain', 'ip', 'hash_md5', 'hash_sha1', 'hash_sha256', 'filename']),
  iocValue: z.string().min(1)
});

router.post('/check', async (req: Request, res: Response) => {
  try {
    const { iocType, iocValue } = checkIOCSchema.parse(req.body);
    
    const iocMatcher = req.app.get('iocMatcher') as IOCMatcherService;
    const match = await iocMatcher.matchIOC(iocType, iocValue);
    
    if (match) {
      return res.json({
        found: true,
        ioc: match
      });
    }
    
    return res.json({
      found: false
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({ error: error.errors });
    }
    res.status(500).json({ error: 'Internal server error' });
  }
});

router.get('/feeds', async (req: Request, res: Response) => {
  // Return list of configured feeds
  res.json({
    feeds: [
      { name: 'MISP', enabled: true },
      { name: 'OTX', enabled: true }
    ]
  });
});

export default router;
```

## Deliverables Checklist

- [ ] MISP client implemented
- [ ] OTX client implemented
- [ ] IOC manager service implemented
- [ ] IOC matcher with bloom filter implemented
- [ ] Sync service implemented
- [ ] Scheduled sync jobs configured
- [ ] API routes created
- [ ] Database migrations for IOC tables
- [ ] Tests written

## Next Steps

After completing Phase 7:
1. Configure MISP and OTX API keys
2. Test feed synchronization
3. Verify IOC matching performance
4. Proceed to Phase 8: Continuous Learning Pipeline
