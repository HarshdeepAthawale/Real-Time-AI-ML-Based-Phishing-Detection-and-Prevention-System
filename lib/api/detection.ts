import { apiPost } from '../api-client';
import { DetectionRequest, DetectionResult } from '../types/api';

/**
 * Detect phishing in email content
 */
export async function detectEmail(request: DetectionRequest): Promise<DetectionResult> {
  return apiPost<DetectionResult>('/api/v1/detect/email', request);
}

/**
 * Detect phishing in URL
 */
export async function detectURL(request: DetectionRequest): Promise<DetectionResult> {
  return apiPost<DetectionResult>('/api/v1/detect/url', request);
}

/**
 * Detect phishing in text content
 */
export async function detectText(request: DetectionRequest): Promise<DetectionResult> {
  return apiPost<DetectionResult>('/api/v1/detect/text', request);
}
