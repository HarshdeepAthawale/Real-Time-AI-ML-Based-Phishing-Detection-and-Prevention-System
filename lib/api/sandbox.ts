import { apiGet, apiPost, getApiBaseUrl } from '../api-client';

export interface SandboxAnalysis {
  analysis_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  analysis_type: 'file' | 'url';
  submitted_at: string;
  started_at?: string;
  completed_at?: string;
  sandbox_provider?: string;
  sandbox_job_id?: string;
  results?: any;
  threat_id?: string;
  threat?: {
    id: string;
    threat_type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    confidence_score: number;
  };
}

export interface SandboxAnalysisList {
  analyses: Array<{
    analysis_id: string;
    status: string;
    analysis_type: string;
    submitted_at: string;
    completed_at?: string;
    threat_id?: string;
  }>;
  pagination: {
    page: number;
    limit: number;
    total: number;
    total_pages: number;
  };
}

export interface SandboxSubmissionResponse {
  analysis_id: string;
  status: string;
  message: string;
}

export interface SandboxStatus {
  sandboxEnabled: boolean;
  provider: string;
  message?: string;
}

/**
 * Get sandbox service status (enabled/disabled)
 */
export async function getSandboxStatus(): Promise<SandboxStatus> {
  try {
    return await apiGet<SandboxStatus>('/api/v1/sandbox/status');
  } catch {
    return { sandboxEnabled: false, provider: 'unknown', message: 'Unable to reach sandbox service' };
  }
}

/**
 * Submit a file for sandbox analysis
 */
export async function submitFileForAnalysis(file: File): Promise<SandboxSubmissionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  // Use axios directly for file uploads to handle FormData properly
  const axios = (await import('axios')).default;
  const apiKey = typeof window !== 'undefined' ? localStorage.getItem('api_key') : null;
  const headers: Record<string, string> = {};
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  
  const response = await axios.post<SandboxSubmissionResponse>(
    `${getApiBaseUrl()}/api/v1/sandbox/analyze/file`,
    formData,
    {
      headers,
      timeout: 60000, // Longer timeout for file uploads
    }
  );
  
  return response.data;
}

/**
 * Submit a URL for sandbox analysis
 */
export async function submitURLForAnalysis(url: string): Promise<SandboxSubmissionResponse> {
  return apiPost<SandboxSubmissionResponse>('/api/v1/sandbox/analyze/url', { url });
}

/**
 * Get sandbox analysis status and results
 */
export async function getSandboxAnalysis(analysisId: string): Promise<SandboxAnalysis> {
  return apiGet<SandboxAnalysis>(`/api/v1/sandbox/analysis/${analysisId}`);
}

/**
 * List sandbox analyses with pagination
 */
export async function listSandboxAnalyses(page: number = 1, limit: number = 20): Promise<SandboxAnalysisList> {
  return apiGet<SandboxAnalysisList>(`/api/v1/sandbox/analyses?page=${page}&limit=${limit}`);
}
