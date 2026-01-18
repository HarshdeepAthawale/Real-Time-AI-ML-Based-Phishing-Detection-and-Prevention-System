import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import axios from 'axios';
import { logger } from './utils/logger';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3003;

// Detection API URL
const DETECTION_API_URL = process.env.DETECTION_API_URL || 'http://detection-api:3001';

app.use(helmet());
app.use(cors());
app.use(express.json());

interface ExtensionCheckRequest {
  url: string;
  page_text?: string;
  page_title?: string;
  screenshot?: string; // base64 encoded
}

interface ExtensionCheckResponse {
  is_phishing: boolean;
  confidence: number;
  warning_message?: string;
  details: any;
  timestamp: string;
}

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'extension-api' });
});

app.post('/api/v1/extension/check', async (req, res) => {
  try {
    const request: ExtensionCheckRequest = req.body;
    
    if (!request.url) {
      return res.status(400).json({ error: 'URL is required' });
    }
    
    // Call detection API
    const detectionRequest: any = {
      url: request.url
    };
    
    if (request.page_text) {
      detectionRequest.text = request.page_text;
    }
    
    if (request.screenshot) {
      detectionRequest.image = request.screenshot;
    }
    
    let detectionResult;
    try {
      const response = await axios.post(`${DETECTION_API_URL}/api/v1/detect`, detectionRequest);
      detectionResult = response.data;
    } catch (error: any) {
      logger.error(`Detection API error: ${error.message}`);
      // Return safe default if detection fails
      detectionResult = {
        is_phishing: false,
        confidence: 0,
        sources: {}
      };
    }
    
    const response: ExtensionCheckResponse = {
      is_phishing: detectionResult.is_phishing || false,
      confidence: detectionResult.overall_confidence || detectionResult.confidence || 0,
      warning_message: detectionResult.is_phishing 
        ? 'This page may be a phishing attempt. Proceed with caution.' 
        : undefined,
      details: {
        url_analysis: detectionResult.sources?.url,
        text_analysis: detectionResult.sources?.nlp,
        visual_analysis: detectionResult.sources?.visual
      },
      timestamp: new Date().toISOString()
    };
    
    res.json(response);
  } catch (error: any) {
    logger.error(`Extension check error: ${error.message}`);
    res.status(500).json({ error: 'Extension check failed', message: error.message });
  }
});

app.post('/api/v1/extension/report', async (req, res) => {
  try {
    const { url, reason, description } = req.body;
    
    // Report to threat intelligence service
    try {
      const threatIntelUrl = process.env.THREAT_INTEL_URL || 'http://threat-intel:3002';
      await axios.post(`${threatIntelUrl}/api/v1/intelligence/report`, {
        identifier: url,
        threat_type: 'phishing',
        confidence: 0.8,
        description: `${reason}: ${description}`
      });
    } catch (error: any) {
      logger.error(`Failed to report to threat intel: ${error.message}`);
    }
    
    logger.info(`Phishing reported via extension: ${url}`);
    res.json({ success: true, message: 'Report submitted successfully' });
  } catch (error: any) {
    logger.error(`Report error: ${error.message}`);
    res.status(500).json({ error: 'Report failed', message: error.message });
  }
});

app.listen(PORT, () => {
  logger.info(`Extension API service running on port ${PORT}`);
});
