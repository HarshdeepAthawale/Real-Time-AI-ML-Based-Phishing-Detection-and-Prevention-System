/**
 * IOC Domain Model (DTO)
 * Used for API requests/responses and internal processing
 */

export type IOCType = 
  | 'url' 
  | 'domain' 
  | 'ip' 
  | 'email'
  | 'hash_md5' 
  | 'hash_sha1' 
  | 'hash_sha256' 
  | 'filename';

export type Severity = 'critical' | 'high' | 'medium' | 'low';

export interface IOC {
  id?: string;
  feedId?: string | null;
  iocType: IOCType;
  iocValue: string;
  threatType?: string | null;
  severity?: Severity | null;
  confidence?: number | null;
  firstSeenAt?: Date | null;
  lastSeenAt?: Date | null;
  source?: string;
  sourceReports?: number;
  metadata?: Record<string, any>;
  createdAt?: Date;
  updatedAt?: Date;
}

export interface IOCCheckRequest {
  iocType: IOCType;
  iocValue: string;
}

export interface IOCCheckResponse {
  found: boolean;
  ioc?: IOC;
  confidence?: number;
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
  iocType?: IOCType;
  severity?: Severity;
  threatType?: string;
  limit?: number;
  offset?: number;
  search?: string;
}

export interface IOCStats {
  total: number;
  byType: Record<IOCType, number>;
  bySeverity: Record<Severity, number>;
  byFeed: Record<string, number>;
  recentCount: number; // Last 24 hours
}
