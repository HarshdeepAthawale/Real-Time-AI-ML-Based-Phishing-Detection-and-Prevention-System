import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import axios from 'axios';
import FormData from 'form-data';
import { logger } from './utils/logger';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Service URLs
const NLP_SERVICE_URL = process.env.NLP_SERVICE_URL || 'http://nlp-service:8000';
const URL_SERVICE_URL = process.env.URL_SERVICE_URL || 'http://url-service:8000';
const VISUAL_SERVICE_URL = process.env.VISUAL_SERVICE_URL || 'http://visual-service:8000';

app.use(helmet());
app.use(cors());
app.use(express.json());

interface DetectionRequest {
  text?: string;
  url?: string;
  image?: string; // base64 encoded image
  include_features?: boolean;
}

interface DetectionResponse {
  is_phishing: boolean;
  confidence: number;
  sources: {
    nlp?: any;
    url?: any;
    visual?: any;
  };
  overall_confidence: number;
  timestamp: string;
}

async function analyzeWithNLPService(text: string, includeFeatures: boolean = false) {
  try {
    const response = await axios.post(`${NLP_SERVICE_URL}/analyze`, {
      text,
      include_features: includeFeatures
    });
    return response.data;
  } catch (error: any) {
    logger.error(`NLP service error: ${error.message}`);
    return null;
  }
}

async function analyzeWithURLService(url: string) {
  try {
    const response = await axios.post(`${URL_SERVICE_URL}/analyze`, {
      url
    });
    return response.data;
  } catch (error: any) {
    logger.error(`URL service error: ${error.message}`);
    return null;
  }
}

async function analyzeWithVisualService(imageData: string) {
  try {
    // Convert base64 to buffer for multipart/form-data
    const buffer = Buffer.from(imageData, 'base64');
    const formData = new FormData();
    formData.append('file', buffer, { filename: 'image.jpg' });
    
    const response = await axios.post(`${VISUAL_SERVICE_URL}/analyze`, formData, {
      headers: formData.getHeaders()
    });
    return response.data;
  } catch (error: any) {
    logger.error(`Visual service error: ${error.message}`);
    return null;
  }
}

interface SourceResult {
  service: 'nlp' | 'url' | 'visual';
  confidence: number;
}

function calculateOverallConfidence(sources: SourceResult[]): number {
  const validSources = sources.filter(s => s !== null && s.confidence !== undefined);
  if (validSources.length === 0) return 0;
  
  // Weighted average (can be adjusted based on service reliability)
  const weights: Record<string, number> = { nlp: 0.4, url: 0.4, visual: 0.2 };
  let weightedSum = 0;
  let totalWeight = 0;
  
  validSources.forEach(source => {
    const weight = weights[source.service] || 0.33;
    weightedSum += source.confidence * weight;
    totalWeight += weight;
  });
  
  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'detection-api' });
});

app.post('/api/v1/detect', async (req, res) => {
  try {
    const request: DetectionRequest = req.body;
    const sources: any = {};
    
    // Analyze with NLP service if text provided
    if (request.text) {
      const nlpResult = await analyzeWithNLPService(request.text, request.include_features);
      if (nlpResult) {
        sources.nlp = { ...nlpResult, service: 'nlp' };
      }
    }
    
    // Analyze with URL service if URL provided
    if (request.url) {
      const urlResult = await analyzeWithURLService(request.url);
      if (urlResult) {
        sources.url = { ...urlResult, service: 'url' };
      }
    }
    
    // Analyze with Visual service if image provided
    if (request.image) {
      const visualResult = await analyzeWithVisualService(request.image);
      if (visualResult) {
        sources.visual = { ...visualResult, service: 'visual' };
      }
    }
    
    // Calculate overall confidence
    const sourceArray: SourceResult[] = Object.values(sources)
      .filter((s): s is SourceResult => 
        s !== null && 
        typeof s === 'object' && 
        'service' in s && 
        'confidence' in s &&
        (s.service === 'nlp' || s.service === 'url' || s.service === 'visual')
      );
    const overallConfidence = calculateOverallConfidence(sourceArray);
    const isPhishing = overallConfidence >= 0.5;
    
    const response: DetectionResponse = {
      is_phishing: isPhishing,
      confidence: overallConfidence,
      sources,
      overall_confidence: overallConfidence,
      timestamp: new Date().toISOString()
    };
    
    res.json(response);
  } catch (error: any) {
    logger.error(`Detection error: ${error.message}`);
    res.status(500).json({ error: 'Detection failed', message: error.message });
  }
});

app.listen(PORT, () => {
  logger.info(`Detection API service running on port ${PORT}`);
});
