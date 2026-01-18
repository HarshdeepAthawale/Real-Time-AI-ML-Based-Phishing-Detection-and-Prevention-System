# Phase 10: Sandbox Integration & Advanced Threat Analysis

## Objective
Integrate dynamic sandbox environments for behavioral analysis of links and attachments, build result processing pipeline, and correlate sandbox results with detection signals.

## Prerequisites
- Phase 6 completed (Detection API)
- Access to sandbox service (Cuckoo, Any.run, or custom)
- File analysis capabilities

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         Sandbox Service                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   File       │  │   Sandbox     │  │   Result     │  │
│  │   Analyzer   │→ │   Submitter  │→ │   Processor  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Behavioral │  │   Correlation│  │   Job Queue  │  │
│  │   Analyzer   │  │   Engine     │  │   Manager    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
backend/core-services/sandbox-service/
├── src/
│   ├── index.ts
│   ├── integrations/
│   │   ├── cuckoo.client.ts         # Cuckoo Sandbox client
│   │   ├── anyrun.client.ts         # Any.run client
│   │   └── base-sandbox.client.ts   # Base sandbox interface
│   ├── services/
│   │   ├── file-analyzer.service.ts # File analysis
│   │   ├── sandbox-submitter.service.ts # Submit to sandbox
│   │   ├── result-processor.service.ts # Process results
│   │   ├── behavioral-analyzer.service.ts # Behavioral analysis
│   │   └── correlation.service.ts   # Correlate with detections
│   ├── jobs/
│   │   └── sandbox-queue.job.ts     # Job queue processing
│   └── models/
│       └── sandbox-analysis.model.ts
├── tests/
├── Dockerfile
├── package.json
└── README.md
```

## Implementation Steps

### 1. Dependencies

**File**: `backend/core-services/sandbox-service/package.json`

```json
{
  "dependencies": {
    "express": "^4.18.2",
    "axios": "^1.6.2",
    "bullmq": "^5.1.0",
    "file-type": "^18.7.0",
    "pdf-parse": "^1.1.1",
    "mammoth": "^1.6.0",
    "ioredis": "^5.3.2",
    "pg": "^8.11.3",
    "winston": "^3.11.0"
  }
}
```

### 2. Base Sandbox Client Interface

**File**: `backend/core-services/sandbox-service/src/integrations/base-sandbox.client.ts`

```typescript
export interface SandboxResult {
  jobId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  analysisId?: string;
  results?: {
    network?: NetworkActivity[];
    filesystem?: FileSystemActivity[];
    processes?: ProcessActivity[];
    registry?: RegistryActivity[];
    screenshots?: string[];
    signatures?: string[];
    score?: number;
  };
  error?: string;
}

export interface NetworkActivity {
  protocol: string;
  destination: string;
  port: number;
  method?: string;
  path?: string;
  statusCode?: number;
}

export interface FileSystemActivity {
  action: 'created' | 'modified' | 'deleted' | 'read';
  path: string;
  fileType?: string;
}

export interface ProcessActivity {
  name: string;
  pid: number;
  commandLine: string;
  parentPid?: number;
}

export interface RegistryActivity {
  action: 'created' | 'modified' | 'deleted';
  key: string;
  value?: string;
}

export abstract class BaseSandboxClient {
  abstract submitFile(fileBuffer: Buffer, filename: string, options?: any): Promise<string>;
  abstract submitURL(url: string, options?: any): Promise<string>;
  abstract getStatus(jobId: string): Promise<SandboxResult>;
  abstract getResults(analysisId: string): Promise<SandboxResult>;
}
```

### 3. Cuckoo Sandbox Client

**File**: `backend/core-services/sandbox-service/src/integrations/cuckoo.client.ts`

```typescript
import axios, { AxiosInstance } from 'axios';
import { BaseSandboxClient, SandboxResult } from './base-sandbox.client';
import { logger } from '../utils/logger';

export class CuckooClient extends BaseSandboxClient {
  private client: AxiosInstance;
  private baseURL: string;
  
  constructor(baseURL: string, apiKey?: string) {
    super();
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL: `${baseURL}/api`,
      headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}
    });
  }
  
  async submitFile(fileBuffer: Buffer, filename: string, options?: any): Promise<string> {
    try {
      const formData = new FormData();
      const blob = new Blob([fileBuffer]);
      formData.append('file', blob, filename);
      
      if (options?.timeout) {
        formData.append('timeout', options.timeout.toString());
      }
      
      const response = await this.client.post('/tasks/create/file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      return response.data.task_id.toString();
    } catch (error) {
      logger.error('Cuckoo file submission failed', error);
      throw error;
    }
  }
  
  async submitURL(url: string, options?: any): Promise<string> {
    try {
      const params: any = { url };
      if (options?.timeout) {
        params.timeout = options.timeout;
      }
      
      const response = await this.client.post('/tasks/create/url', params);
      return response.data.task_id.toString();
    } catch (error) {
      logger.error('Cuckoo URL submission failed', error);
      throw error;
    }
  }
  
  async getStatus(jobId: string): Promise<SandboxResult> {
    try {
      const response = await this.client.get(`/tasks/view/${jobId}`);
      const task = response.data.task;
      
      return {
        jobId,
        status: this.mapStatus(task.status),
        analysisId: task.id?.toString()
      };
    } catch (error) {
      logger.error('Cuckoo status check failed', error);
      throw error;
    }
  }
  
  async getResults(analysisId: string): Promise<SandboxResult> {
    try {
      const report = await this.client.get(`/tasks/report/${analysisId}/json`);
      const data = report.data;
      
      return {
        jobId: analysisId,
        status: 'completed',
        analysisId,
        results: {
          network: this.extractNetworkActivity(data.network),
          filesystem: this.extractFileSystemActivity(data.target?.file),
          processes: this.extractProcessActivity(data.processes),
          screenshots: data.screenshots?.map((s: any) => s.path),
          signatures: data.signatures?.map((s: any) => s.name),
          score: data.info?.score || 0
        }
      };
    } catch (error) {
      logger.error('Cuckoo results fetch failed', error);
      throw error;
    }
  }
  
  private mapStatus(status: string): 'pending' | 'running' | 'completed' | 'failed' {
    const mapping: Record<string, 'pending' | 'running' | 'completed' | 'failed'> = {
      'pending': 'pending',
      'running': 'running',
      'completed': 'completed',
      'reported': 'completed',
      'failed': 'failed'
    };
    return mapping[status] || 'pending';
  }
  
  private extractNetworkActivity(network: any): NetworkActivity[] {
    if (!network) return [];
    
    return network.map((item: any) => ({
      protocol: item.protocol || 'tcp',
      destination: item.dst,
      port: item.dport,
      method: item.method,
      path: item.uri,
      statusCode: item.status_code
    }));
  }
  
  private extractFileSystemActivity(file: any): FileSystemActivity[] {
    if (!file) return [];
    
    return [{
      action: 'created',
      path: file.path,
      fileType: file.type
    }];
  }
  
  private extractProcessActivity(processes: any[]): ProcessActivity[] {
    if (!processes) return [];
    
    return processes.map((proc: any) => ({
      name: proc.process_name,
      pid: proc.pid,
      commandLine: proc.command_line,
      parentPid: proc.parent_id
    }));
  }
}
```

### 4. File Analyzer Service

**File**: `backend/core-services/sandbox-service/src/services/file-analyzer.service.ts`

```typescript
import fileType from 'file-type';
import pdfParse from 'pdf-parse';
import mammoth from 'mammoth';
import { logger } from '../utils/logger';

export interface FileAnalysis {
  filename: string;
  fileType: string;
  mimeType: string;
  size: number;
  hash: {
    md5: string;
    sha1: string;
    sha256: string;
  };
  metadata: Record<string, any>;
  extractedText?: string;
  isExecutable: boolean;
  requiresSandbox: boolean;
}

export class FileAnalyzerService {
  async analyzeFile(fileBuffer: Buffer, filename: string): Promise<FileAnalysis> {
    const type = await fileType.fromBuffer(fileBuffer);
    const mimeType = type?.mime || 'application/octet-stream';
    const fileTypeStr = type?.ext || 'unknown';
    
    // Calculate hashes
    const crypto = require('crypto');
    const md5 = crypto.createHash('md5').update(fileBuffer).digest('hex');
    const sha1 = crypto.createHash('sha1').update(fileBuffer).digest('hex');
    const sha256 = crypto.createHash('sha256').update(fileBuffer).digest('hex');
    
    // Extract metadata and text based on file type
    let metadata: Record<string, any> = {};
    let extractedText: string | undefined;
    
    try {
      if (mimeType === 'application/pdf') {
        const pdfData = await pdfParse(fileBuffer);
        extractedText = pdfData.text;
        metadata = {
          pages: pdfData.numpages,
          info: pdfData.info
        };
      } else if (mimeType.includes('wordprocessingml') || mimeType.includes('msword')) {
        const result = await mammoth.extractRawText({ buffer: fileBuffer });
        extractedText = result.value;
      }
    } catch (error) {
      logger.warn('Failed to extract text from file', error);
    }
    
    // Determine if file requires sandbox analysis
    const executableTypes = [
      'application/x-msdownload',
      'application/x-executable',
      'application/x-msdos-program',
      'application/x-dosexec'
    ];
    const isExecutable = executableTypes.includes(mimeType) || 
                        ['.exe', '.dll', '.bat', '.cmd', '.scr', '.com'].some(ext => 
                          filename.toLowerCase().endsWith(ext));
    
    const requiresSandbox = isExecutable || 
                          mimeType.includes('javascript') ||
                          mimeType.includes('script');
    
    return {
      filename,
      fileType: fileTypeStr,
      mimeType,
      size: fileBuffer.length,
      hash: { md5, sha1, sha256 },
      metadata,
      extractedText,
      isExecutable,
      requiresSandbox
    };
  }
}
```

### 5. Sandbox Submitter Service

**File**: `backend/core-services/sandbox-service/src/services/sandbox-submitter.service.ts`

```typescript
import { BaseSandboxClient } from '../integrations/base-sandbox.client';
import { FileAnalyzerService, FileAnalysis } from './file-analyzer.service';
import { Pool } from 'pg';
import { logger } from '../utils/logger';

export class SandboxSubmitterService {
  private sandboxClient: BaseSandboxClient;
  private fileAnalyzer: FileAnalyzerService;
  private pool: Pool;
  
  constructor(
    sandboxClient: BaseSandboxClient,
    fileAnalyzer: FileAnalyzerService,
    pool: Pool
  ) {
    this.sandboxClient = sandboxClient;
    this.fileAnalyzer = fileAnalyzer;
    this.pool = pool;
  }
  
  async submitFile(
    fileBuffer: Buffer,
    filename: string,
    organizationId?: string
  ): Promise<string> {
    // Analyze file first
    const analysis = await this.fileAnalyzer.analyzeFile(fileBuffer, filename);
    
    // Check if file requires sandbox analysis
    if (!analysis.requiresSandbox) {
      logger.info(`File ${filename} does not require sandbox analysis`);
      // Still create record but mark as not requiring sandbox
      return await this.createAnalysisRecord(analysis, null, organizationId);
    }
    
    // Submit to sandbox
    const jobId = await this.sandboxClient.submitFile(fileBuffer, filename, {
      timeout: 300 // 5 minutes
    });
    
    // Create analysis record
    const analysisId = await this.createAnalysisRecord(analysis, jobId, organizationId);
    
    logger.info(`File submitted to sandbox: ${jobId}`);
    return analysisId;
  }
  
  async submitURL(url: string, organizationId?: string): Promise<string> {
    // Submit URL to sandbox
    const jobId = await this.sandboxClient.submitURL(url, {
      timeout: 300
    });
    
    // Create analysis record
    const query = `
      INSERT INTO sandbox_analyses (
        organization_id, analysis_type, target_url,
        sandbox_provider, sandbox_job_id, status
      ) VALUES ($1, $2, $3, $4, $5, $6)
      RETURNING id
    `;
    
    const result = await this.pool.query(query, [
      organizationId,
      'url',
      url,
      'cuckoo', // or detect from client
      jobId,
      'pending'
    ]);
    
    return result.rows[0].id;
  }
  
  private async createAnalysisRecord(
    fileAnalysis: FileAnalysis,
    jobId: string | null,
    organizationId?: string
  ): Promise<string> {
    const query = `
      INSERT INTO sandbox_analyses (
        organization_id, analysis_type, target_file_hash,
        sandbox_provider, sandbox_job_id, status, result_data
      ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING id
    `;
    
    const result = await this.pool.query(query, [
      organizationId,
      'file',
      fileAnalysis.hash.sha256,
      'cuckoo',
      jobId,
      jobId ? 'pending' : 'not_required',
      JSON.stringify({
        filename: fileAnalysis.filename,
        fileType: fileAnalysis.fileType,
        mimeType: fileAnalysis.mimeType,
        size: fileAnalysis.size,
        hash: fileAnalysis.hash,
        metadata: fileAnalysis.metadata
      })
    ]);
    
    return result.rows[0].id;
  }
}
```

### 6. Result Processor Service

**File**: `backend/core-services/sandbox-service/src/services/result-processor.service.ts`

```typescript
import { BaseSandboxClient, SandboxResult } from '../integrations/base-sandbox.client';
import { Pool } from 'pg';
import { BehavioralAnalyzerService } from './behavioral-analyzer.service';
import { logger } from '../utils/logger';

export class ResultProcessorService {
  private sandboxClient: BaseSandboxClient;
  private pool: Pool;
  private behavioralAnalyzer: BehavioralAnalyzerService;
  
  constructor(
    sandboxClient: BaseSandboxClient,
    pool: Pool,
    behavioralAnalyzer: BehavioralAnalyzerService
  ) {
    this.sandboxClient = sandboxClient;
    this.pool = pool;
    this.behavioralAnalyzer = behavioralAnalyzer;
  }
  
  async processResults(analysisId: string): Promise<void> {
    try {
      // Get analysis record
      const analysis = await this.getAnalysisRecord(analysisId);
      
      if (!analysis.sandbox_job_id) {
        logger.warn(`Analysis ${analysisId} has no sandbox job ID`);
        return;
      }
      
      // Get sandbox results
      const sandboxResult = await this.sandboxClient.getResults(analysis.sandbox_job_id);
      
      // Analyze behavioral indicators
      const behavioralAnalysis = this.behavioralAnalyzer.analyze(sandboxResult);
      
      // Update analysis record
      await this.updateAnalysisRecord(analysisId, {
        status: sandboxResult.status,
        resultData: {
          sandbox: sandboxResult.results,
          behavioral: behavioralAnalysis,
          isMalicious: behavioralAnalysis.isMalicious,
          threatScore: behavioralAnalysis.threatScore
        }
      });
      
      // If malicious, create threat record
      if (behavioralAnalysis.isMalicious) {
        await this.createThreatRecord(analysisId, behavioralAnalysis);
      }
      
    } catch (error) {
      logger.error(`Failed to process results for analysis ${analysisId}`, error);
      await this.updateAnalysisRecord(analysisId, {
        status: 'failed',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  }
  
  private async getAnalysisRecord(analysisId: string): Promise<any> {
    const query = 'SELECT * FROM sandbox_analyses WHERE id = $1';
    const result = await this.pool.query(query, [analysisId]);
    return result.rows[0];
  }
  
  private async updateAnalysisRecord(
    analysisId: string,
    updates: {
      status?: string;
      resultData?: any;
      error?: string;
    }
  ): Promise<void> {
    const query = `
      UPDATE sandbox_analyses
      SET status = COALESCE($2, status),
          result_data = COALESCE($3, result_data),
          updated_at = CURRENT_TIMESTAMP
      WHERE id = $1
    `;
    
    await this.pool.query(query, [
      analysisId,
      updates.status,
      updates.resultData ? JSON.stringify(updates.resultData) : null
    ]);
  }
  
  private async createThreatRecord(analysisId: string, behavioralAnalysis: any): Promise<void> {
    // Create threat record in threats table
    const query = `
      INSERT INTO threats (
        organization_id, threat_type, severity,
        source, source_value, confidence_score,
        metadata
      ) VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING id
    `;
    
    // Implementation depends on your threat record structure
  }
}
```

### 7. Behavioral Analyzer Service

**File**: `backend/core-services/sandbox-service/src/services/behavioral-analyzer.service.ts`

```typescript
import { SandboxResult } from '../integrations/base-sandbox.client';

export interface BehavioralAnalysis {
  isMalicious: boolean;
  threatScore: number;
  indicators: string[];
  networkActivity: {
    suspiciousConnections: number;
    c2Connections: number;
    dataExfiltration: boolean;
  };
  fileSystemActivity: {
    suspiciousModifications: number;
    systemFileAccess: boolean;
  };
  processActivity: {
    suspiciousProcesses: number;
    processInjection: boolean;
  };
}

export class BehavioralAnalyzerService {
  analyze(sandboxResult: SandboxResult): BehavioralAnalysis {
    const results = sandboxResult.results || {};
    const indicators: string[] = [];
    let threatScore = 0;
    
    // Analyze network activity
    const networkAnalysis = this.analyzeNetworkActivity(results.network || []);
    if (networkAnalysis.suspiciousConnections > 0) {
      indicators.push('suspicious_network_activity');
      threatScore += 20;
    }
    if (networkAnalysis.c2Connections > 0) {
      indicators.push('c2_communication');
      threatScore += 40;
    }
    if (networkAnalysis.dataExfiltration) {
      indicators.push('data_exfiltration');
      threatScore += 30;
    }
    
    // Analyze file system activity
    const fsAnalysis = this.analyzeFileSystemActivity(results.filesystem || []);
    if (fsAnalysis.suspiciousModifications > 0) {
      indicators.push('suspicious_file_modifications');
      threatScore += 15;
    }
    if (fsAnalysis.systemFileAccess) {
      indicators.push('system_file_access');
      threatScore += 25;
    }
    
    // Analyze process activity
    const processAnalysis = this.analyzeProcessActivity(results.processes || []);
    if (processAnalysis.suspiciousProcesses > 0) {
      indicators.push('suspicious_processes');
      threatScore += 20;
    }
    if (processAnalysis.processInjection) {
      indicators.push('process_injection');
      threatScore += 35;
    }
    
    // Check sandbox score
    if (results.score && results.score > 7) {
      indicators.push('high_sandbox_score');
      threatScore += 30;
    }
    
    // Check signatures
    if (results.signatures && results.signatures.length > 0) {
      indicators.push('malware_signatures_detected');
      threatScore += 25;
    }
    
    const isMalicious = threatScore >= 50;
    
    return {
      isMalicious,
      threatScore: Math.min(100, threatScore),
      indicators,
      networkActivity: networkAnalysis,
      fileSystemActivity: fsAnalysis,
      processActivity: processAnalysis
    };
  }
  
  private analyzeNetworkActivity(network: any[]): {
    suspiciousConnections: number;
    c2Connections: number;
    dataExfiltration: boolean;
  } {
    // Suspicious ports
    const suspiciousPorts = [4444, 5555, 6666, 8080, 8443];
    const suspiciousConnections = network.filter(conn => 
      suspiciousPorts.includes(conn.port)
    ).length;
    
    // C2 indicators (known malicious domains/IPs)
    const c2Connections = network.filter(conn =>
      this.isKnownC2Domain(conn.destination)
    ).length;
    
    // Data exfiltration (large outbound transfers)
    const dataExfiltration = network.some(conn =>
      conn.method === 'POST' && conn.statusCode === 200
    );
    
    return {
      suspiciousConnections,
      c2Connections,
      dataExfiltration
    };
  }
  
  private analyzeFileSystemActivity(filesystem: any[]): {
    suspiciousModifications: number;
    systemFileAccess: boolean;
  } {
    const systemPaths = ['C:\\Windows\\System32', '/etc', '/usr/bin'];
    const suspiciousModifications = filesystem.filter(fs =>
      fs.action === 'modified' || fs.action === 'created'
    ).length;
    
    const systemFileAccess = filesystem.some(fs =>
      systemPaths.some(path => fs.path.includes(path))
    );
    
    return {
      suspiciousModifications,
      systemFileAccess
    };
  }
  
  private analyzeProcessActivity(processes: any[]): {
    suspiciousProcesses: number;
    processInjection: boolean;
  } {
    const suspiciousNames = ['cmd.exe', 'powershell.exe', 'wscript.exe'];
    const suspiciousProcesses = processes.filter(proc =>
      suspiciousNames.some(name => proc.name.toLowerCase().includes(name))
    ).length;
    
    const processInjection = processes.some(proc =>
      proc.parentPid && proc.parentPid !== 1 // Not root process
    );
    
    return {
      suspiciousProcesses,
      processInjection
    };
  }
  
  private isKnownC2Domain(domain: string): boolean {
    // Check against known C2 domain list
    // This would query your IOC database
    return false;
  }
}
```

### 8. Job Queue Processing

**File**: `backend/core-services/sandbox-service/src/jobs/sandbox-queue.job.ts`

```typescript
import { Queue, Worker } from 'bullmq';
import Redis from 'ioredis';
import { ResultProcessorService } from '../services/result-processor.service';
import { logger } from '../utils/logger';

export class SandboxQueueJob {
  private queue: Queue;
  private worker: Worker;
  private resultProcessor: ResultProcessorService;
  
  constructor(redis: Redis, resultProcessor: ResultProcessorService) {
    this.resultProcessor = resultProcessor;
    
    this.queue = new Queue('sandbox-analysis', {
      connection: redis
    });
    
    this.worker = new Worker('sandbox-analysis', async (job) => {
      const { analysisId } = job.data;
      await this.resultProcessor.processResults(analysisId);
    }, {
      connection: redis,
      concurrency: 5 // Process 5 analyses concurrently
    });
    
    this.setupEventHandlers();
  }
  
  private setupEventHandlers(): void {
    this.worker.on('completed', (job) => {
      logger.info(`Sandbox analysis completed: ${job.id}`);
    });
    
    this.worker.on('failed', (job, err) => {
      logger.error(`Sandbox analysis failed: ${job?.id}`, err);
    });
  }
  
  async addAnalysisJob(analysisId: string, delay: number = 0): Promise<void> {
    await this.queue.add('process-analysis', {
      analysisId
    }, {
      delay: delay * 1000, // Delay in milliseconds
      attempts: 3,
      backoff: {
        type: 'exponential',
        delay: 5000
      }
    });
  }
}
```

## Deliverables Checklist

- [ ] Base sandbox client interface
- [ ] Cuckoo Sandbox integration
- [ ] Any.run integration (optional)
- [ ] File analyzer service
- [ ] Sandbox submitter service
- [ ] Result processor service
- [ ] Behavioral analyzer service
- [ ] Job queue for sandbox processing
- [ ] API endpoints for sandbox submission
- [ ] Correlation with detection API
- [ ] Tests written

## Next Steps

After completing Phase 10:
1. Configure sandbox service
2. Test file and URL submissions
3. Verify behavioral analysis accuracy
4. Integrate with detection API
5. **System Integration**: Connect all phases together
6. **Frontend Integration**: Connect Next.js frontend to backend APIs

## Final Integration Steps

After all 10 phases are complete:
1. End-to-end testing
2. Performance optimization
3. Security audit
4. Documentation
5. Deployment to production
