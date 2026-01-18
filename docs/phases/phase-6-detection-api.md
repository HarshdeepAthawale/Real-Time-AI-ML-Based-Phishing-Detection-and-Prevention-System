# Phase 6: Real-time Detection API (Node.js/TypeScript)

## Objective
Build a high-performance API service that orchestrates ML services for real-time threat detection with sub-50ms latency, WebSocket support for event streaming, and an ensemble decision engine.

## Prerequisites
- Phases 1-5 completed
- All ML services deployed and accessible
- Node.js 20+ and TypeScript installed

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         Detection API (Node.js/TypeScript)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Request   │  │   ML Service │  │   Decision   │  │
│  │  Router     │→ │ Orchestrator │→ │   Engine     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Cache      │  │   WebSocket   │  │   Event      │  │
│  │   Manager    │  │   Server      │  │   Streamer    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
backend/core-services/detection-api/
├── src/
│   ├── index.ts                    # Application entry point
│   ├── config/
│   │   └── index.ts                # Configuration
│   ├── routes/
│   │   ├── detection.routes.ts    # Detection endpoints
│   │   └── websocket.routes.ts    # WebSocket endpoints
│   ├── services/
│   │   ├── orchestrator.service.ts # ML service orchestration
│   │   ├── decision-engine.service.ts # Threat decision logic
│   │   ├── cache.service.ts       # Redis caching
│   │   └── event-streamer.service.ts # Event streaming
│   ├── middleware/
│   │   ├── auth.middleware.ts     # Authentication
│   │   ├── rate-limit.middleware.ts # Rate limiting
│   │   └── error-handler.middleware.ts # Error handling
│   ├── models/
│   │   └── detection.model.ts     # Data models
│   ├── utils/
│   │   ├── logger.ts              # Logging utility
│   │   └── validators.ts          # Input validation
│   └── types/
│       └── index.ts               # TypeScript types
├── tests/
├── Dockerfile
├── package.json
├── tsconfig.json
└── README.md
```

## Implementation Steps

### 1. Dependencies

**File**: `backend/core-services/detection-api/package.json`

```json
{
  "name": "detection-api",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.2",
    "fastify": "^4.24.3",
    "socket.io": "^4.5.4",
    "axios": "^1.6.2",
    "ioredis": "^5.3.2",
    "bullmq": "^5.1.0",
    "zod": "^3.22.4",
    "dotenv": "^16.3.1",
    "winston": "^3.11.0",
    "express-rate-limit": "^7.1.5",
    "helmet": "^7.1.0",
    "cors": "^2.8.5",
    "compression": "^1.7.4"
  },
  "devDependencies": {
    "@types/node": "^20.10.0",
    "@types/express": "^4.17.21",
    "typescript": "^5.3.3",
    "ts-node": "^10.9.2",
    "nodemon": "^3.0.2",
    "@typescript-eslint/eslint-plugin": "^6.13.1",
    "jest": "^29.7.0",
    "@types/jest": "^29.5.8"
  }
}
```

### 2. Application Setup

**File**: `backend/core-services/detection-api/src/index.ts`

```typescript
import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import { config } from './config';
import { logger } from './utils/logger';
import { errorHandler } from './middleware/error-handler.middleware';
import detectionRoutes from './routes/detection.routes';
import { setupWebSocket } from './routes/websocket.routes';
import { CacheService } from './services/cache.service';
import { EventStreamerService } from './services/event-streamer.service';

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: config.cors.origins,
    methods: ['GET', 'POST']
  }
});

// Middleware
app.use(helmet());
app.use(cors(config.cors));
app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Initialize services
const cacheService = new CacheService();
const eventStreamer = new EventStreamerService(io);

// Routes
app.use('/api/v1/detect', detectionRoutes);

// WebSocket setup
setupWebSocket(io, eventStreamer);

// Health check
app.get('/health', async (req, res) => {
  const cacheStatus = await cacheService.isConnected();
  res.json({
    status: 'healthy',
    cache: cacheStatus ? 'connected' : 'disconnected',
    timestamp: new Date().toISOString()
  });
});

// Error handling
app.use(errorHandler);

// Start server
const PORT = config.port || 3000;
httpServer.listen(PORT, () => {
  logger.info(`Detection API server running on port ${PORT}`);
});
```

### 3. ML Service Orchestrator

**File**: `backend/core-services/detection-api/src/services/orchestrator.service.ts`

```typescript
import axios, { AxiosInstance } from 'axios';
import { config } from '../config';
import { logger } from '../utils/logger';
import { DetectionRequest, MLServiceResponse } from '../types';

export class OrchestratorService {
  private nlpClient: AxiosInstance;
  private urlClient: AxiosInstance;
  private visualClient: AxiosInstance;
  
  constructor() {
    this.nlpClient = axios.create({
      baseURL: config.mlServices.nlp,
      timeout: 5000
    });
    
    this.urlClient = axios.create({
      baseURL: config.mlServices.url,
      timeout: 10000
    });
    
    this.visualClient = axios.create({
      baseURL: config.mlServices.visual,
      timeout: 30000
    });
  }
  
  async analyzeEmail(request: DetectionRequest): Promise<MLServiceResponse> {
    const startTime = Date.now();
    
    try {
      // Parallel calls to NLP and URL services
      const [nlpResult, urlResults] = await Promise.all([
        this.nlpClient.post('/api/v1/analyze-email', {
          raw_email: request.emailContent,
          include_features: true
        }),
        this.extractAndAnalyzeURLs(request.emailContent)
      ]);
      
      const processingTime = Date.now() - startTime;
      
      return {
        nlp: nlpResult.data,
        url: urlResults,
        visual: null, // Visual analysis is async for emails
        processingTimeMs: processingTime
      };
    } catch (error) {
      logger.error('Error in email analysis orchestration', error);
      throw error;
    }
  }
  
  async analyzeURL(request: DetectionRequest): Promise<MLServiceResponse> {
    const startTime = Date.now();
    
    try {
      // Parallel calls to URL and Visual services
      const [urlResult, visualResult] = await Promise.all([
        this.urlClient.post('/api/v1/analyze-url', {
          url: request.url,
          legitimate_domain: request.legitimateDomain
        }),
        this.visualClient.post('/api/v1/analyze-page', {
          url: request.url,
          legitimate_url: request.legitimateUrl
        }).catch(err => {
          // Visual analysis can fail, don't block URL analysis
          logger.warn('Visual analysis failed', err);
          return { data: null };
        })
      ]);
      
      // Extract text from URL page for NLP analysis
      const pageText = visualResult.data?.dom_analysis?.text || '';
      let nlpResult = null;
      
      if (pageText) {
        try {
          nlpResult = await this.nlpClient.post('/api/v1/analyze-text', {
            text: pageText,
            include_features: false
          });
        } catch (err) {
          logger.warn('NLP analysis failed', err);
        }
      }
      
      const processingTime = Date.now() - startTime;
      
      return {
        nlp: nlpResult?.data || null,
        url: urlResult.data,
        visual: visualResult.data,
        processingTimeMs: processingTime
      };
    } catch (error) {
      logger.error('Error in URL analysis orchestration', error);
      throw error;
    }
  }
  
  private async extractAndAnalyzeURLs(emailContent: string): Promise<any[]> {
    // Extract URLs from email
    const urlPattern = /https?:\/\/[^\s<>"{}|\\^`\[\]]+/g;
    const urls = emailContent.match(urlPattern) || [];
    
    // Analyze each URL (limit to first 5 for performance)
    const urlAnalyses = await Promise.allSettled(
      urls.slice(0, 5).map(url => 
        this.urlClient.post('/api/v1/analyze-url', { url })
      )
    );
    
    return urlAnalyses
      .filter(result => result.status === 'fulfilled')
      .map(result => (result as PromiseFulfilledResult<any>).value.data);
  }
}
```

### 4. Decision Engine

**File**: `backend/core-services/detection-api/src/services/decision-engine.service.ts`

```typescript
import { MLServiceResponse } from '../types';
import { Threat, ThreatSeverity, ThreatType } from '../models/detection.model';

export class DecisionEngineService {
  private readonly PHISHING_THRESHOLD = 0.7;
  private readonly HIGH_CONFIDENCE_THRESHOLD = 0.85;
  
  makeDecision(mlResponse: MLServiceResponse, input: any): Threat {
    const scores = this.calculateScores(mlResponse);
    const ensembleScore = this.calculateEnsembleScore(scores);
    const severity = this.determineSeverity(ensembleScore, scores);
    const threatType = this.determineThreatType(mlResponse, scores);
    
    const isThreat = ensembleScore >= this.PHISHING_THRESHOLD;
    const confidence = this.calculateConfidence(scores);
    
    return {
      isThreat,
      confidence,
      severity,
      threatType,
      scores: {
        ensemble: ensembleScore,
        nlp: scores.nlp,
        url: scores.url,
        visual: scores.visual
      },
      indicators: this.extractIndicators(mlResponse),
      metadata: {
        processingTimeMs: mlResponse.processingTimeMs,
        timestamp: new Date().toISOString()
      }
    };
  }
  
  private calculateScores(mlResponse: MLServiceResponse): {
    nlp: number;
    url: number;
    visual: number;
  } {
    const scores = {
      nlp: 0,
      url: 0,
      visual: 0
    };
    
    // NLP score
    if (mlResponse.nlp) {
      scores.nlp = mlResponse.nlp.phishing_probability || 0;
      
      // Boost score based on urgency and AI-generated content
      if (mlResponse.nlp.urgency_score > 70) {
        scores.nlp += 0.1;
      }
      if (mlResponse.nlp.ai_generated_probability > 0.7) {
        scores.nlp += 0.15;
      }
      scores.nlp = Math.min(1.0, scores.nlp);
    }
    
    // URL score
    if (mlResponse.url) {
      const urlData = mlResponse.url;
      
      // Combine multiple URL analysis results
      if (urlData.domain_analysis?.is_suspicious) {
        scores.url += 0.3;
      }
      if (urlData.whois_analysis?.is_suspicious) {
        scores.url += 0.2;
      }
      if (urlData.redirect_analysis?.is_suspicious) {
        scores.url += 0.25;
      }
      if (urlData.homoglyph_analysis?.is_suspicious) {
        scores.url += 0.25;
      }
      
      scores.url = Math.min(1.0, scores.url);
    }
    
    // Visual score
    if (mlResponse.visual) {
      const visualData = mlResponse.visual;
      
      if (visualData.form_analysis?.is_suspicious) {
        scores.visual += 0.4;
      }
      if (visualData.brand_prediction?.is_brand_impersonation) {
        scores.visual += 0.3;
      }
      if (visualData.similarity_analysis?.is_similar && 
          visualData.similarity_analysis?.similarity_score < 0.9) {
        scores.visual += 0.2; // Similar but not identical = suspicious
      }
      
      scores.visual = Math.min(1.0, scores.visual);
    }
    
    return scores;
  }
  
  private calculateEnsembleScore(scores: { nlp: number; url: number; visual: number }): number {
    // Weighted ensemble
    const weights = {
      nlp: 0.4,
      url: 0.4,
      visual: 0.2
    };
    
    // Only include scores that are available
    let totalWeight = 0;
    let weightedSum = 0;
    
    if (scores.nlp > 0) {
      weightedSum += scores.nlp * weights.nlp;
      totalWeight += weights.nlp;
    }
    
    if (scores.url > 0) {
      weightedSum += scores.url * weights.url;
      totalWeight += weights.url;
    }
    
    if (scores.visual > 0) {
      weightedSum += scores.visual * weights.visual;
      totalWeight += weights.visual;
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : 0;
  }
  
  private determineSeverity(
    ensembleScore: number,
    scores: { nlp: number; url: number; visual: number }
  ): ThreatSeverity {
    if (ensembleScore >= 0.9) {
      return 'critical';
    } else if (ensembleScore >= 0.75) {
      return 'high';
    } else if (ensembleScore >= 0.6) {
      return 'medium';
    } else {
      return 'low';
    }
  }
  
  private determineThreatType(
    mlResponse: MLServiceResponse,
    scores: { nlp: number; url: number; visual: number }
  ): ThreatType {
    if (scores.visual > 0.7) {
      return 'brand_impersonation';
    } else if (scores.url > 0.7) {
      return 'url_spoofing';
    } else if (mlResponse.nlp?.ai_generated_probability > 0.7) {
      return 'ai_generated';
    } else {
      return 'email_phishing';
    }
  }
  
  private calculateConfidence(scores: { nlp: number; url: number; visual: number }): number {
    // Confidence is based on agreement between models
    const nonZeroScores = Object.values(scores).filter(s => s > 0);
    if (nonZeroScores.length === 0) return 0;
    
    const avgScore = nonZeroScores.reduce((a, b) => a + b, 0) / nonZeroScores.length;
    const variance = nonZeroScores.reduce((sum, score) => {
      return sum + Math.pow(score - avgScore, 2);
    }, 0) / nonZeroScores.length;
    
    // Lower variance = higher confidence
    const confidence = Math.max(0, 1 - variance);
    return confidence;
  }
  
  private extractIndicators(mlResponse: MLServiceResponse): string[] {
    const indicators: string[] = [];
    
    if (mlResponse.nlp?.urgency_score > 70) {
      indicators.push('high_urgency_language');
    }
    if (mlResponse.nlp?.ai_generated_probability > 0.7) {
      indicators.push('ai_generated_content');
    }
    if (mlResponse.url?.homoglyph_analysis?.is_suspicious) {
      indicators.push('homoglyph_attack');
    }
    if (mlResponse.url?.redirect_analysis?.redirect_count > 3) {
      indicators.push('excessive_redirects');
    }
    if (mlResponse.visual?.form_analysis?.is_suspicious) {
      indicators.push('credential_harvesting_form');
    }
    if (mlResponse.visual?.brand_prediction?.is_brand_impersonation) {
      indicators.push('brand_impersonation');
    }
    
    return indicators;
  }
}
```

### 5. Cache Service

**File**: `backend/core-services/detection-api/src/services/cache.service.ts`

```typescript
import Redis from 'ioredis';
import { config } from '../config';
import { logger } from '../utils/logger';
import crypto from 'crypto';

export class CacheService {
  private client: Redis;
  
  constructor() {
    this.client = new Redis({
      host: config.redis.host,
      port: config.redis.port,
      password: config.redis.password,
      retryStrategy: (times) => {
        const delay = Math.min(times * 50, 2000);
        return delay;
      }
    });
    
    this.client.on('error', (err) => {
      logger.error('Redis error', err);
    });
  }
  
  async get(key: string): Promise<any | null> {
    try {
      const value = await this.client.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      logger.error('Cache get error', error);
      return null;
    }
  }
  
  async set(key: string, value: any, ttlSeconds: number = 3600): Promise<void> {
    try {
      await this.client.setex(key, ttlSeconds, JSON.stringify(value));
    } catch (error) {
      logger.error('Cache set error', error);
    }
  }
  
  async getOrSet<T>(
    key: string,
    fetcher: () => Promise<T>,
    ttlSeconds: number = 3600
  ): Promise<T> {
    const cached = await this.get(key);
    if (cached !== null) {
      return cached as T;
    }
    
    const value = await fetcher();
    await this.set(key, value, ttlSeconds);
    return value;
  }
  
  generateCacheKey(type: string, input: string): string {
    const hash = crypto.createHash('sha256').update(input).digest('hex');
    return `${type}:${hash}`;
  }
  
  async isConnected(): Promise<boolean> {
    try {
      await this.client.ping();
      return true;
    } catch {
      return false;
    }
  }
}
```

### 6. Event Streamer Service

**File**: `backend/core-services/detection-api/src/services/event-streamer.service.ts`

```typescript
import { Server, Socket } from 'socket.io';
import { logger } from '../utils/logger';
import { Threat } from '../models/detection.model';

export class EventStreamerService {
  private io: Server;
  private connectedClients: Map<string, Socket> = new Map();
  
  constructor(io: Server) {
    this.io = io;
    this.setupConnectionHandlers();
  }
  
  private setupConnectionHandlers(): void {
    this.io.on('connection', (socket: Socket) => {
      logger.info(`Client connected: ${socket.id}`);
      this.connectedClients.set(socket.id, socket);
      
      socket.on('disconnect', () => {
        logger.info(`Client disconnected: ${socket.id}`);
        this.connectedClients.delete(socket.id);
      });
      
      socket.on('subscribe', (organizationId: string) => {
        socket.join(`org:${organizationId}`);
        logger.info(`Client ${socket.id} subscribed to org ${organizationId}`);
      });
    });
  }
  
  broadcastThreat(organizationId: string, threat: Threat): void {
    this.io.to(`org:${organizationId}`).emit('threat_detected', {
      ...threat,
      timestamp: new Date().toISOString()
    });
  }
  
  broadcastEvent(eventType: string, data: any): void {
    this.io.emit(eventType, {
      ...data,
      timestamp: new Date().toISOString()
    });
  }
}
```

### 7. Detection Routes

**File**: `backend/core-services/detection-api/src/routes/detection.routes.ts`

```typescript
import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { OrchestratorService } from '../services/orchestrator.service';
import { DecisionEngineService } from '../services/decision-engine.service';
import { CacheService } from '../services/cache.service';
import { EventStreamerService } from '../services/event-streamer.service';
import { authMiddleware } from '../middleware/auth.middleware';
import { rateLimitMiddleware } from '../middleware/rate-limit.middleware';

const router = Router();
const orchestrator = new OrchestratorService();
const decisionEngine = new DecisionEngineService();
const cacheService = new CacheService();
let eventStreamer: EventStreamerService;

export function setEventStreamer(streamer: EventStreamerService): void {
  eventStreamer = streamer;
}

const detectEmailSchema = z.object({
  emailContent: z.string().min(1),
  organizationId: z.string().uuid().optional()
});

const detectURLSchema = z.object({
  url: z.string().url(),
  legitimateDomain: z.string().optional(),
  legitimateUrl: z.string().url().optional()
});

router.post('/email', 
  authMiddleware,
  rateLimitMiddleware,
  async (req: Request, res: Response) => {
    try {
      const validated = detectEmailSchema.parse(req.body);
      
      // Check cache
      const cacheKey = cacheService.generateCacheKey('email', validated.emailContent);
      const cached = await cacheService.get(cacheKey);
      
      if (cached) {
        return res.json({
          ...cached,
          cached: true
        });
      }
      
      // Analyze
      const mlResponse = await orchestrator.analyzeEmail({
        emailContent: validated.emailContent
      });
      
      const threat = decisionEngine.makeDecision(mlResponse, validated);
      
      // Cache result
      await cacheService.set(cacheKey, threat, 3600);
      
      // Broadcast event
      if (eventStreamer && validated.organizationId) {
        eventStreamer.broadcastThreat(validated.organizationId, threat);
      }
      
      res.json(threat);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ error: error.errors });
      }
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

router.post('/url',
  authMiddleware,
  rateLimitMiddleware,
  async (req: Request, res: Response) => {
    try {
      const validated = detectURLSchema.parse(req.body);
      
      // Check cache
      const cacheKey = cacheService.generateCacheKey('url', validated.url);
      const cached = await cacheService.get(cacheKey);
      
      if (cached) {
        return res.json({
          ...cached,
          cached: true
        });
      }
      
      // Analyze
      const mlResponse = await orchestrator.analyzeURL({
        url: validated.url,
        legitimateDomain: validated.legitimateDomain,
        legitimateUrl: validated.legitimateUrl
      });
      
      const threat = decisionEngine.makeDecision(mlResponse, validated);
      
      // Cache result
      await cacheService.set(cacheKey, threat, 7200); // URLs cached longer
      
      // Broadcast event
      if (eventStreamer) {
        eventStreamer.broadcastEvent('url_analyzed', {
          url: validated.url,
          threat
        });
      }
      
      res.json(threat);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ error: error.errors });
      }
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

export default router;
```

### 8. WebSocket Routes

**File**: `backend/core-services/detection-api/src/routes/websocket.routes.ts`

```typescript
import { Server, Socket } from 'socket.io';
import { EventStreamerService } from '../services/event-streamer.service';

export function setupWebSocket(
  io: Server,
  eventStreamer: EventStreamerService
): void {
  io.on('connection', (socket: Socket) => {
    socket.on('subscribe', (organizationId: string) => {
      socket.join(`org:${organizationId}`);
    });
    
    socket.on('unsubscribe', (organizationId: string) => {
      socket.leave(`org:${organizationId}`);
    });
  });
}
```

## Deliverables Checklist

- [ ] Express/Fastify application created
- [ ] ML service orchestrator implemented
- [ ] Decision engine implemented
- [ ] Cache service implemented
- [ ] WebSocket server implemented
- [ ] Event streaming implemented
- [ ] Authentication middleware
- [ ] Rate limiting middleware
- [ ] API routes created
- [ ] Error handling
- [ ] Docker configuration
- [ ] Tests written

## Performance Targets

- Detection latency: <50ms (cached), <100ms (new)
- Throughput: 1000+ requests/second
- WebSocket connections: 10,000+ concurrent
- Cache hit rate: >80%

## Next Steps

After completing Phase 6:
1. Deploy detection API
2. Load test the service
3. Monitor performance metrics
4. Proceed to Phase 7: Threat Intelligence Integration
