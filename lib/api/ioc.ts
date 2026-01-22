import { apiGet, apiPost } from '../api-client';

export interface IOCCheckRequest {
  iocType: 'url' | 'domain' | 'ip' | 'hash_md5' | 'hash_sha1' | 'hash_sha256' | 'filename' | 'email';
  iocValue: string;
}

export interface IOCCheckResponse {
  found: boolean;
  ioc?: {
    iocValue: string;
    iocType: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    threatType?: string;
    confidence?: number;
  };
  confidence?: number;
  enrichment?: any;
}

export interface IOCBulkCheckRequest {
  iocs: IOCCheckRequest[];
}

export interface IOCBulkCheckResponse {
  results: IOCCheckResponse[];
  summary: {
    total: number;
    found: number;
    notFound: number;
  };
}

export interface IOCSearchParams {
  iocType?: string;
  severity?: 'critical' | 'high' | 'medium' | 'low';
  threatType?: string;
  search?: string;
  limit?: number;
  offset?: number;
}

export interface IOCSearchResponse {
  iocs: Array<{
    iocValue: string;
    iocType: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    threatType?: string;
    confidence?: number;
    firstSeenAt?: string;
    lastSeenAt?: string;
  }>;
  total: number;
  limit: number;
  offset: number;
}

export interface IOCReportRequest {
  iocType: 'url' | 'domain' | 'ip' | 'hash_md5' | 'hash_sha1' | 'hash_sha256' | 'filename' | 'email';
  iocValue: string;
  threatType?: string;
  severity?: 'critical' | 'high' | 'medium' | 'low';
  confidence?: number;
  metadata?: Record<string, any>;
}

export interface IOCStats {
  total: number;
  byType: Record<string, number>;
  bySeverity: Record<string, number>;
}

/**
 * Check a single IOC
 */
export async function checkIOC(
  request: IOCCheckRequest,
  enrich: boolean = false
): Promise<IOCCheckResponse> {
  return apiPost<IOCCheckResponse>(
    `/api/v1/ioc/check?enrich=${enrich}`,
    request
  );
}

/**
 * Bulk check IOCs
 */
export async function bulkCheckIOC(
  request: IOCBulkCheckRequest
): Promise<IOCBulkCheckResponse> {
  return apiPost<IOCBulkCheckResponse>('/api/v1/ioc/bulk-check', request);
}

/**
 * Search IOCs
 */
export async function searchIOCs(params: IOCSearchParams): Promise<IOCSearchResponse> {
  const queryParams = new URLSearchParams();
  if (params.iocType) queryParams.append('iocType', params.iocType);
  if (params.severity) queryParams.append('severity', params.severity);
  if (params.threatType) queryParams.append('threatType', params.threatType);
  if (params.search) queryParams.append('search', params.search);
  if (params.limit) queryParams.append('limit', params.limit.toString());
  if (params.offset) queryParams.append('offset', params.offset.toString());

  return apiGet<IOCSearchResponse>(`/api/v1/ioc/search?${queryParams.toString()}`);
}

/**
 * Report a new IOC
 */
export async function reportIOC(request: IOCReportRequest): Promise<any> {
  return apiPost('/api/v1/ioc/report', request);
}

/**
 * Get IOC statistics
 */
export async function getIOCStats(): Promise<IOCStats> {
  return apiGet<IOCStats>('/api/v1/ioc/stats');
}
