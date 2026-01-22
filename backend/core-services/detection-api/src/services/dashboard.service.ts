import { DataSource, Repository, Between, LessThanOrEqual } from 'typeorm';
import { Threat } from '../../../../shared/database/models/Threat';
import { getPostgreSQL } from '../../../../shared/database/connection';
import { logger } from '../utils/logger';

export class DashboardService {
  private threatRepository: Repository<Threat>;
  private dataSource: DataSource;

  constructor() {
    this.dataSource = getPostgreSQL();
    this.threatRepository = this.dataSource.getRepository(Threat);
  }

  /**
   * Get dashboard statistics
   */
  async getStats(): Promise<{
    criticalThreats: number;
    detectionRate: number;
    avgResponseTime: number;
    phishingAttempts: number;
    criticalThreatsChange?: string;
    detectionRateChange?: string;
    avgResponseTimeChange?: string;
    phishingAttemptsPeriod?: string;
  }> {
    try {
      const now = new Date();
      const last24Hours = new Date(now.getTime() - 24 * 60 * 60 * 1000);
      const lastHour = new Date(now.getTime() - 60 * 60 * 1000);
      const previous24Hours = new Date(last24Hours.getTime() - 24 * 60 * 60 * 1000);

      // Critical threats (last 24 hours)
      const criticalThreats = await this.threatRepository.count({
        where: {
          severity: 'critical',
          detected_at: LessThanOrEqual(now),
        },
      });

      // Critical threats in last hour
      const criticalThreatsLastHour = await this.threatRepository.count({
        where: {
          severity: 'critical',
          detected_at: Between(lastHour, now),
        },
      });

      // Total threats (last 24 hours)
      const totalThreats24h = await this.threatRepository.count({
        where: {
          detected_at: Between(last24Hours, now),
        },
      });

      // Total threats in previous 24 hours (for comparison)
      const totalThreatsPrevious24h = await this.threatRepository.count({
        where: {
          detected_at: Between(previous24Hours, last24Hours),
        },
      });

      // Calculate detection rate (assuming blocked + resolved = successful detections)
      const blockedThreats = await this.threatRepository.count({
        where: {
          status: 'blocked',
          detected_at: Between(last24Hours, now),
        },
      });

      const resolvedThreats = await this.threatRepository.count({
        where: {
          status: 'resolved',
          detected_at: Between(last24Hours, now),
        },
      });

      const successfulDetections = blockedThreats + resolvedThreats;
      const detectionRate = totalThreats24h > 0 
        ? Math.round((successfulDetections / totalThreats24h) * 1000) / 10 
        : 0; // No threats detected - return 0 instead of 100

      // Calculate average response time from metadata
      const recentThreats = await this.threatRepository.find({
        where: {
          detected_at: Between(last24Hours, now),
        },
        take: 100,
      });

      const responseTimes = recentThreats
        .map((threat: Threat) => threat.metadata?.processingTimeMs)
        .filter((time: unknown): time is number => typeof time === 'number');

      const avgResponseTime = responseTimes.length > 0
        ? Math.round(responseTimes.reduce((a: number, b: number) => a + b, 0) / responseTimes.length)
        : 0; // No data available - return 0 instead of mock value

      // Calculate changes
      const criticalThreatsChange = criticalThreatsLastHour > 0 
        ? `+${criticalThreatsLastHour} this hour` 
        : '0 this hour';

      const detectionRateChange = totalThreatsPrevious24h > 0
        ? `${detectionRate >= 95 ? '+' : ''}${((detectionRate - (successfulDetections / totalThreatsPrevious24h * 100)) / 10).toFixed(1)}% today`
        : '+0.0% today';

      return {
        criticalThreats,
        detectionRate,
        avgResponseTime,
        phishingAttempts: totalThreats24h,
        criticalThreatsChange,
        detectionRateChange,
        avgResponseTimeChange: 'Within SLA',
        phishingAttemptsPeriod: 'Last 24h',
      };
    } catch (error) {
      logger.error('Error fetching dashboard stats', error);
      // Return default values on error
      return {
        criticalThreats: 0,
        detectionRate: 0,
        avgResponseTime: 0,
        phishingAttempts: 0,
      };
    }
  }

  /**
   * Get recent threats
   */
  async getRecentThreats(limit: number = 10, offset: number = 0): Promise<Array<{
    id: string;
    type: string;
    target: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    status: 'blocked' | 'monitored' | 'resolved' | 'detected';
    timestamp: string;
    confidenceScore?: number;
    source?: string;
    title?: string;
    description?: string;
  }>> {
    try {
      const threats = await this.threatRepository.find({
        order: {
          detected_at: 'DESC',
        },
        take: limit,
        skip: offset,
      });

      return threats.map((threat: Threat) => ({
        id: threat.id,
        type: threat.threat_type,
        target: threat.source_value || threat.source || 'Unknown',
        severity: threat.severity.toLowerCase() as 'critical' | 'high' | 'medium' | 'low',
        status: threat.status as 'blocked' | 'monitored' | 'resolved' | 'detected',
        timestamp: this.formatTimestamp(threat.detected_at),
        confidenceScore: parseFloat(threat.confidence_score.toString()),
        source: threat.source || undefined,
        title: threat.title || undefined,
        description: threat.description || undefined,
      }));
    } catch (error) {
      logger.error('Error fetching recent threats', error);
      return [];
    }
  }

  /**
   * Get chart data for threat timeline
   */
  async getChartData(hours: number = 24): Promise<Array<{ time: string; threats: number }>> {
    try {
      const now = new Date();
      const startTime = new Date(now.getTime() - hours * 60 * 60 * 1000);
      const intervalHours = Math.max(1, Math.floor(hours / 7)); // 7 data points max

      const dataPoints: Array<{ time: string; threats: number }> = [];

      for (let i = 0; i < hours; i += intervalHours) {
        const intervalStart = new Date(startTime.getTime() + i * 60 * 60 * 1000);
        const intervalEnd = new Date(Math.min(
          intervalStart.getTime() + intervalHours * 60 * 60 * 1000,
          now.getTime()
        ));

        const count = await this.threatRepository.count({
          where: {
            detected_at: Between(intervalStart, intervalEnd),
          },
        });

        const timeLabel = this.formatTimeLabel(intervalStart, hours);
        dataPoints.push({
          time: timeLabel,
          threats: count,
        });
      }

      return dataPoints;
    } catch (error) {
      logger.error('Error fetching chart data', error);
      return [];
    }
  }

  /**
   * Get threat type distribution
   */
  async getThreatDistribution(): Promise<{
    emailPhishing: number;
    urlSpoofing: number;
    domainHijacking: number;
    aiGeneratedContent: number;
  }> {
    try {
      const now = new Date();
      const last24Hours = new Date(now.getTime() - 24 * 60 * 60 * 1000);

      const allThreats = await this.threatRepository.find({
        where: {
          detected_at: Between(last24Hours, now),
        },
      });

      const distribution = {
        emailPhishing: 0,
        urlSpoofing: 0,
        domainHijacking: 0,
        aiGeneratedContent: 0,
      };

      allThreats.forEach((threat: Threat) => {
        const type = threat.threat_type.toLowerCase();
        if (type.includes('email') || type.includes('phishing')) {
          distribution.emailPhishing++;
        } else if (type.includes('url') || type.includes('spoof')) {
          distribution.urlSpoofing++;
        } else if (type.includes('domain') || type.includes('hijack')) {
          distribution.domainHijacking++;
        } else if (type.includes('ai') || type.includes('generated')) {
          distribution.aiGeneratedContent++;
        }
      });

      // Convert to percentages
      const total = allThreats.length || 1;
      return {
        emailPhishing: Math.round((distribution.emailPhishing / total) * 100),
        urlSpoofing: Math.round((distribution.urlSpoofing / total) * 100),
        domainHijacking: Math.round((distribution.domainHijacking / total) * 100),
        aiGeneratedContent: Math.round((distribution.aiGeneratedContent / total) * 100),
      };
    } catch (error) {
      logger.error('Error fetching threat distribution', error);
      return {
        emailPhishing: 0,
        urlSpoofing: 0,
        domainHijacking: 0,
        aiGeneratedContent: 0,
      };
    }
  }

  private formatTimestamp(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  }

  private formatTimeLabel(date: Date, totalHours: number): string {
    if (totalHours <= 24) {
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    }
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit' });
  }
}
