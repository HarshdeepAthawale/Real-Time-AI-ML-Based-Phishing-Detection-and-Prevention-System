import { DataSource, Between, MoreThan } from 'typeorm';
import { Threat } from '../../../../shared/database/models/Threat';
import { Detection } from '../../../../shared/database/models/Detection';
import { SandboxAnalysis } from '../../../../shared/database/models/SandboxAnalysis';
import { getPostgreSQL } from '../../../../shared/database/connection';
import { logger } from '../utils/logger';

export interface TimelineData {
  date: string;
  threats: number;
  detections: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
}

export interface ThreatDistribution {
  by_severity: Array<{ severity: string; count: number; percentage: number }>;
  by_type: Array<{ type: string; count: number; percentage: number }>;
  by_source: Array<{ source: string; count: number; percentage: number }>;
}

export interface PerformanceMetrics {
  avg_detection_time_ms: number;
  avg_ml_score: number;
  avg_threat_intel_matches: number;
  total_urls_analyzed: number;
  total_emails_analyzed: number;
  total_sandbox_analyses: number;
  cache_hit_rate: number;
}

export interface TrendAnalysis {
  period: string;
  threat_trend: 'increasing' | 'decreasing' | 'stable';
  change_percentage: number;
  predictions: Array<{ date: string; predicted_threats: number }>;
}

export class AnalyticsService {
  private dataSource: DataSource;

  constructor() {
    this.dataSource = getPostgreSQL();
  }

  /**
   * Get timeline data for charts
   */
  async getTimelineData(
    organizationId: string,
    startDate: Date,
    endDate: Date,
    interval: 'hour' | 'day' | 'week' | 'month' = 'day'
  ): Promise<TimelineData[]> {
    let dateFormat: string;
    switch (interval) {
      case 'hour':
        dateFormat = 'YYYY-MM-DD HH24:00:00';
        break;
      case 'week':
        dateFormat = 'IYYY-IW';
        break;
      case 'month':
        dateFormat = 'YYYY-MM';
        break;
      default:
        dateFormat = 'YYYY-MM-DD';
    }

    const result = await this.dataSource.query(`
      SELECT 
        TO_CHAR(created_at, $3) as date,
        COUNT(*) FILTER (WHERE true) as threats,
        COUNT(DISTINCT detection_id) as detections,
        COUNT(*) FILTER (WHERE severity = 'critical') as critical,
        COUNT(*) FILTER (WHERE severity = 'high') as high,
        COUNT(*) FILTER (WHERE severity = 'medium') as medium,
        COUNT(*) FILTER (WHERE severity = 'low') as low
      FROM threats
      WHERE organization_id = $1
        AND created_at BETWEEN $2 AND $4
      GROUP BY TO_CHAR(created_at, $3)
      ORDER BY date ASC
    `, [organizationId, startDate, dateFormat, endDate]);

    return result;
  }

  /**
   * Get threat distribution statistics
   */
  async getThreatDistribution(organizationId: string): Promise<ThreatDistribution> {
    // Distribution by severity
    const severityResult = await this.dataSource.query(`
      SELECT 
        severity,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER(), 0), 2) as percentage
      FROM threats
      WHERE organization_id = $1
      GROUP BY severity
      ORDER BY 
        CASE severity
          WHEN 'critical' THEN 1
          WHEN 'high' THEN 2
          WHEN 'medium' THEN 3
          WHEN 'low' THEN 4
          ELSE 5
        END
    `, [organizationId]);

    // Distribution by type
    const typeResult = await this.dataSource.query(`
      SELECT 
        threat_type as type,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER(), 0), 2) as percentage
      FROM threats
      WHERE organization_id = $1 AND threat_type IS NOT NULL
      GROUP BY threat_type
      ORDER BY count DESC
      LIMIT 10
    `, [organizationId]);

    // Distribution by source
    const sourceResult = await this.dataSource.query(`
      SELECT 
        source,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER(), 0), 2) as percentage
      FROM threats
      WHERE organization_id = $1 AND source IS NOT NULL
      GROUP BY source
      ORDER BY count DESC
      LIMIT 10
    `, [organizationId]);

    return {
      by_severity: severityResult.map((r: any) => ({
        severity: r.severity,
        count: parseInt(r.count),
        percentage: parseFloat(r.percentage || 0)
      })),
      by_type: typeResult.map((r: any) => ({
        type: r.type,
        count: parseInt(r.count),
        percentage: parseFloat(r.percentage || 0)
      })),
      by_source: sourceResult.map((r: any) => ({
        source: r.source,
        count: parseInt(r.count),
        percentage: parseFloat(r.percentage || 0)
      }))
    };
  }

  /**
   * Get performance metrics
   */
  async getPerformanceMetrics(organizationId: string): Promise<PerformanceMetrics> {
    const result = await this.dataSource.query(`
      SELECT 
        AVG(COALESCE((metadata->>'processing_time_ms')::numeric, 0)) as avg_detection_time_ms,
        AVG(confidence) as avg_ml_score,
        AVG(COALESCE((metadata->>'threat_intel_matches')::numeric, 0)) as avg_threat_intel_matches,
        COUNT(*) FILTER (WHERE target LIKE 'http%') as total_urls_analyzed,
        COUNT(*) FILTER (WHERE target LIKE '%@%') as total_emails_analyzed,
        (SELECT COUNT(*) FROM sandbox_analyses WHERE organization_id = $1) as total_sandbox_analyses,
        COALESCE(
          COUNT(*) FILTER (WHERE metadata->>'cache_hit' = 'true') * 100.0 / NULLIF(COUNT(*), 0),
          0
        ) as cache_hit_rate
      FROM detections
      WHERE organization_id = $1
    `, [organizationId]);

    const row = result[0];

    return {
      avg_detection_time_ms: Math.round(parseFloat(row.avg_detection_time_ms || 0)),
      avg_ml_score: Math.round(parseFloat(row.avg_ml_score || 0) * 100) / 100,
      avg_threat_intel_matches: Math.round(parseFloat(row.avg_threat_intel_matches || 0) * 10) / 10,
      total_urls_analyzed: parseInt(row.total_urls_analyzed || 0),
      total_emails_analyzed: parseInt(row.total_emails_analyzed || 0),
      total_sandbox_analyses: parseInt(row.total_sandbox_analyses || 0),
      cache_hit_rate: Math.round(parseFloat(row.cache_hit_rate || 0) * 10) / 10
    };
  }

  /**
   * Get trend analysis with predictions
   */
  async getTrendAnalysis(organizationId: string, days: number = 30): Promise<TrendAnalysis> {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    // Get historical data
    const result = await this.dataSource.query(`
      SELECT 
        DATE(created_at) as date,
        COUNT(*) as count
      FROM threats
      WHERE organization_id = $1
        AND created_at >= $2
      GROUP BY DATE(created_at)
      ORDER BY date ASC
    `, [organizationId, startDate]);

    if (result.length === 0) {
      return {
        period: `${days} days`,
        threat_trend: 'stable',
        change_percentage: 0,
        predictions: []
      };
    }

    // Calculate trend using simple linear regression
    const counts = result.map((r: any) => parseInt(r.count));
    const n = counts.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = counts.reduce((a, b) => a + b, 0);
    const sumXY = counts.reduce((sum, y, x) => sum + x * y, 0);
    const sumXX = (n * (n - 1) * (2 * n - 1)) / 6;

    const slope = n * sumXX === sumX * sumX ? 0 : (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const avgCount = sumY / n;

    // Determine trend
    let trend: 'increasing' | 'decreasing' | 'stable';
    const changePercentage = avgCount === 0 ? 0 : (slope / avgCount) * 100;

    if (changePercentage > 5) trend = 'increasing';
    else if (changePercentage < -5) trend = 'decreasing';
    else trend = 'stable';

    // Generate predictions for next 7 days
    const predictions = [];
    const lastCount = counts[counts.length - 1];
    for (let i = 1; i <= 7; i++) {
      const predictedCount = Math.max(0, Math.round(lastCount + slope * i));
      const predictedDate = new Date();
      predictedDate.setDate(predictedDate.getDate() + i);
      predictions.push({
        date: predictedDate.toISOString().split('T')[0],
        predicted_threats: predictedCount
      });
    }

    return {
      period: `${days} days`,
      threat_trend: trend,
      change_percentage: Math.round(changePercentage * 100) / 100,
      predictions
    };
  }

  /**
   * Get top threats (most frequent)
   */
  async getTopThreats(organizationId: string, limit: number = 10): Promise<Array<{
    target: string;
    count: number;
    last_seen: Date;
    severity: string;
  }>> {
    const result = await this.dataSource.query(`
      SELECT 
        target,
        COUNT(*) as count,
        MAX(created_at) as last_seen,
        MODE() WITHIN GROUP (ORDER BY severity) as severity
      FROM threats
      WHERE organization_id = $1
      GROUP BY target
      ORDER BY count DESC
      LIMIT $2
    `, [organizationId, limit]);

    return result;
  }

  /**
   * Get detection accuracy metrics
   */
  async getDetectionAccuracy(organizationId: string): Promise<{
    total_detections: number;
    confirmed_threats: number;
    false_positives: number;
    false_negatives: number;
    accuracy_rate: number;
    precision: number;
    recall: number;
  }> {
    const result = await this.dataSource.query(`
      SELECT 
        COUNT(*) as total_detections,
        COUNT(*) FILTER (WHERE df.feedback_type = 'true_positive') as confirmed_threats,
        COUNT(*) FILTER (WHERE df.feedback_type = 'false_positive') as false_positives,
        COUNT(*) FILTER (WHERE df.feedback_type = 'false_negative') as false_negatives
      FROM detections d
      LEFT JOIN detection_feedback df ON d.id = df.detection_id
      WHERE d.organization_id = $1
    `, [organizationId]);

    const row = result[0];
    const total = parseInt(row.total_detections || 0);
    const tp = parseInt(row.confirmed_threats || 0);
    const fp = parseInt(row.false_positives || 0);
    const fn = parseInt(row.false_negatives || 0);

    const tn = total - tp - fp - fn;
    const accuracy = total > 0 ? ((tp + tn) / total) * 100 : 0;
    const precision = (tp + fp) > 0 ? (tp / (tp + fp)) * 100 : 0;
    const recall = (tp + fn) > 0 ? (tp / (tp + fn)) * 100 : 0;

    return {
      total_detections: total,
      confirmed_threats: tp,
      false_positives: fp,
      false_negatives: fn,
      accuracy_rate: Math.round(accuracy * 100) / 100,
      precision: Math.round(precision * 100) / 100,
      recall: Math.round(recall * 100) / 100
    };
  }

  /**
   * Get hourly threat distribution (heatmap data)
   */
  async getHourlyDistribution(organizationId: string, days: number = 7): Promise<Array<{
    day_of_week: number;
    hour: number;
    count: number;
  }>> {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const result = await this.dataSource.query(`
      SELECT 
        EXTRACT(DOW FROM created_at) as day_of_week,
        EXTRACT(HOUR FROM created_at) as hour,
        COUNT(*) as count
      FROM threats
      WHERE organization_id = $1
        AND created_at >= $2
      GROUP BY day_of_week, hour
      ORDER BY day_of_week, hour
    `, [organizationId, startDate]);

    return result.map((r: any) => ({
      day_of_week: parseInt(r.day_of_week),
      hour: parseInt(r.hour),
      count: parseInt(r.count)
    }));
  }

  /**
   * Get ML model performance comparison
   */
  async getMLModelPerformance(organizationId: string): Promise<Array<{
    model: string;
    avg_score: number;
    avg_processing_time: number;
    total_predictions: number;
  }>> {
    const result = await this.dataSource.query(`
      SELECT 
        'NLP' as model,
        AVG(COALESCE((scores->>'nlp')::numeric, 0)) as avg_score,
        AVG(COALESCE((metadata->>'nlp_time_ms')::numeric, 0)) as avg_processing_time,
        COUNT(*) FILTER (WHERE scores->>'nlp' IS NOT NULL) as total_predictions
      FROM detections
      WHERE organization_id = $1
      UNION ALL
      SELECT 
        'URL' as model,
        AVG(COALESCE((scores->>'url')::numeric, 0)) as avg_score,
        AVG(COALESCE((metadata->>'url_time_ms')::numeric, 0)) as avg_processing_time,
        COUNT(*) FILTER (WHERE scores->>'url' IS NOT NULL) as total_predictions
      FROM detections
      WHERE organization_id = $1
      UNION ALL
      SELECT 
        'Visual' as model,
        AVG(COALESCE((scores->>'visual')::numeric, 0)) as avg_score,
        AVG(COALESCE((metadata->>'visual_time_ms')::numeric, 0)) as avg_processing_time,
        COUNT(*) FILTER (WHERE scores->>'visual' IS NOT NULL) as total_predictions
      FROM detections
      WHERE organization_id = $1
    `, [organizationId]);

    return result.map((r: any) => ({
      model: r.model,
      avg_score: Math.round(parseFloat(r.avg_score || 0) * 100) / 100,
      avg_processing_time: Math.round(parseFloat(r.avg_processing_time || 0)),
      total_predictions: parseInt(r.total_predictions || 0)
    }));
  }
}
