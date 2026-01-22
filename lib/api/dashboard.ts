import { apiGet } from '../api-client';
import { DashboardStats, Threat, ChartDataPoint, ThreatDistribution } from '../types/api';

/**
 * Get dashboard statistics
 */
export async function getDashboardStats(): Promise<DashboardStats> {
  return apiGet<DashboardStats>('/api/v1/dashboard/stats');
}

/**
 * Get recent threats
 * @param limit - Number of threats to return (default: 10)
 * @param offset - Offset for pagination (default: 0)
 */
export async function getRecentThreats(limit: number = 10, offset: number = 0): Promise<Threat[]> {
  return apiGet<Threat[]>(`/api/v1/dashboard/threats?limit=${limit}&offset=${offset}`);
}

/**
 * Get chart data for threat timeline
 * @param hours - Number of hours to look back (default: 24)
 */
export async function getChartData(hours: number = 24): Promise<ChartDataPoint[]> {
  return apiGet<ChartDataPoint[]>(`/api/v1/dashboard/chart?hours=${hours}`);
}

/**
 * Get threat type distribution
 */
export async function getThreatDistribution(): Promise<ThreatDistribution> {
  return apiGet<ThreatDistribution>('/api/v1/dashboard/distribution');
}
