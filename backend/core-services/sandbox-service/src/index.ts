import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import { logger } from './utils/logger';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3004;

app.use(helmet());
app.use(cors());
app.use(express.json());

interface SandboxAnalysisRequest {
  url?: string;
  file_hash?: string;
  file_content?: string; // base64 encoded
  analysis_type: 'url' | 'file' | 'email';
}

interface SandboxAnalysisResponse {
  analysis_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  results?: {
    is_malicious: boolean;
    threat_type?: string;
    confidence: number;
    behaviors?: string[];
    indicators?: any;
  };
  created_at: string;
  completed_at?: string;
}

const analyses: Map<string, SandboxAnalysisResponse> = new Map();

async function analyzeInSandbox(_request: SandboxAnalysisRequest): Promise<SandboxAnalysisResponse> {
  const analysisId = `sandbox-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  
  const analysis: SandboxAnalysisResponse = {
    analysis_id: analysisId,
    status: 'pending',
    created_at: new Date().toISOString()
  };
  
  analyses.set(analysisId, analysis);
  
  // Simulate sandbox analysis (would integrate with actual sandbox like Cuckoo, Joe Sandbox, etc.)
  setTimeout(() => {
    analysis.status = 'running';
    logger.info(`Sandbox analysis started: ${analysisId}`);
    
    // Simulate analysis completion
    setTimeout(() => {
      analysis.status = 'completed';
      analysis.completed_at = new Date().toISOString();
      analysis.results = {
        is_malicious: Math.random() > 0.7, // 30% chance of malicious
        threat_type: 'phishing',
        confidence: 0.75,
        behaviors: ['network_activity', 'file_download'],
        indicators: {
          suspicious_domains: ['example-malicious.com'],
          file_downloads: 1,
          network_connections: 3
        }
      };
      logger.info(`Sandbox analysis completed: ${analysisId}`);
    }, 5000); // Simulate 5 second analysis
  }, 1000);
  
  return analysis;
}

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'sandbox-service' });
});

app.post('/api/v1/sandbox/analyze', async (req, res) => {
  try {
    const request: SandboxAnalysisRequest = req.body;
    
    if (!request.analysis_type) {
      return res.status(400).json({ error: 'analysis_type is required' });
    }
    
    // Start sandbox analysis
    const analysis = await analyzeInSandbox(request);
    
    res.status(202).json(analysis);
  } catch (error: any) {
    logger.error(`Sandbox analysis error: ${error.message}`);
    res.status(500).json({ error: 'Sandbox analysis failed', message: error.message });
  }
});

app.get('/api/v1/sandbox/analysis/:analysisId', (req, res) => {
  try {
    const { analysisId } = req.params;
    const analysis = analyses.get(analysisId);
    
    if (!analysis) {
      return res.status(404).json({ error: 'Analysis not found' });
    }
    
    res.json(analysis);
  } catch (error: any) {
    logger.error(`Get analysis error: ${error.message}`);
    res.status(500).json({ error: 'Failed to get analysis', message: error.message });
  }
});

app.get('/api/v1/sandbox/analyses', (req, res) => {
  try {
    const allAnalyses = Array.from(analyses.values());
    res.json({
      analyses: allAnalyses,
      count: allAnalyses.length
    });
  } catch (error: any) {
    logger.error(`List analyses error: ${error.message}`);
    res.status(500).json({ error: 'Failed to list analyses', message: error.message });
  }
});

app.listen(PORT, () => {
  logger.info(`Sandbox service running on port ${PORT}`);
});
