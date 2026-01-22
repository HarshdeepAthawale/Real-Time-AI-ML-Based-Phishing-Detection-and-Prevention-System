import { apiGet } from '../api-client';
import { MaliciousDomain, ThreatPattern, IOC, ThreatIntelligenceSummary } from '../types/api';

/**
 * Get malicious domains list
 * @param limit - Number of domains to return (default: 50)
 * @param offset - Offset for pagination (default: 0)
 */
export async function getMaliciousDomains(limit: number = 50, offset: number = 0): Promise<MaliciousDomain[]> {
  return apiGet<MaliciousDomain[]>(`/api/v1/intelligence/domains?limit=${limit}&offset=${offset}`);
}

/**
 * Get threat patterns
 * @param limit - Number of patterns to return (default: 20)
 */
export async function getThreatPatterns(limit: number = 20): Promise<ThreatPattern[]> {
  return apiGet<ThreatPattern[]>(`/api/v1/intelligence/patterns?limit=${limit}`);
}

/**
 * Get IOCs (Indicators of Compromise)
 * @param limit - Number of IOCs to return (default: 50)
 * @param offset - Offset for pagination (default: 0)
 * @param type - Optional IOC type filter
 */
export async function getIOCs(limit: number = 50, offset: number = 0, type?: string): Promise<IOC[]> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });
  if (type) {
    params.append('type', type);
  }
  return apiGet<IOC[]>(`/api/v1/intelligence/iocs?${params.toString()}`);
}

/**
 * Get threat intelligence summary statistics
 */
export async function getThreatIntelligenceSummary(): Promise<ThreatIntelligenceSummary> {
  return apiGet<ThreatIntelligenceSummary>('/api/v1/intelligence/summary');
}
