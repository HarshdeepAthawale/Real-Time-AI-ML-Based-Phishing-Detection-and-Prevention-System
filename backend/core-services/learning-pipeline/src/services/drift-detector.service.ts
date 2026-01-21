import { DataSource } from 'typeorm';
import { logger } from '../utils/logger';
import { ModelPerformance, MLModel } from '../../../shared/database/models';

export interface DriftResult {
  hasDrift: boolean;
  driftScore: number;
  metrics: {
    recent: {
      accuracy: number;
      precision: number;
      recall: number;
      f1Score: number;
      falsePositiveRate: number;
    };
    baseline: {
      accuracy: number;
      precision: number;
      recall: number;
      f1Score: number;
      falsePositiveRate: number;
    };
  };
  recommendation?: string;
}

export class DriftDetectorService {
  private dataSource: DataSource;
  private driftThreshold: number;

  constructor(dataSource: DataSource) {
    this.dataSource = dataSource;
    this.driftThreshold = parseFloat(process.env.DRIFT_THRESHOLD || '0.05');
  }

  /**
   * Detect drift for a specific model
   */
  async detectDrift(modelId: string, windowDays: number = 7): Promise<DriftResult> {
    try {
      const performanceRepository = this.dataSource.getRepository(ModelPerformance);

      // Get recent performance metrics
      const recentStartDate = new Date();
      recentStartDate.setDate(recentStartDate.getDate() - windowDays);
      
      const recentMetrics = await performanceRepository
        .createQueryBuilder('perf')
        .select('AVG(perf.accuracy)', 'avg_accuracy')
        .addSelect('AVG(perf.precision)', 'avg_precision')
        .addSelect('AVG(perf.recall)', 'avg_recall')
        .addSelect('AVG(perf.f1_score)', 'avg_f1')
        .addSelect('AVG(perf.false_positive_rate)', 'avg_fpr')
        .where('perf.model_id = :modelId', { modelId })
        .andWhere('perf.date >= :startDate', { startDate: recentStartDate })
        .getRawOne();

      // Get baseline metrics (previous window)
      const baselineEndDate = new Date(recentStartDate);
      const baselineStartDate = new Date();
      baselineStartDate.setDate(baselineStartDate.getDate() - windowDays * 2);

      const baselineMetrics = await performanceRepository
        .createQueryBuilder('perf')
        .select('AVG(perf.accuracy)', 'avg_accuracy')
        .addSelect('AVG(perf.precision)', 'avg_precision')
        .addSelect('AVG(perf.recall)', 'avg_recall')
        .addSelect('AVG(perf.f1_score)', 'avg_f1')
        .addSelect('AVG(perf.false_positive_rate)', 'avg_fpr')
        .where('perf.model_id = :modelId', { modelId })
        .andWhere('perf.date >= :startDate', { startDate: baselineStartDate })
        .andWhere('perf.date < :endDate', { endDate: baselineEndDate })
        .getRawOne();

      if (!recentMetrics || !baselineMetrics) {
        logger.warn(`Insufficient performance data for drift detection (model: ${modelId})`);
        return {
          hasDrift: false,
          driftScore: 0,
          metrics: {
            recent: {
              accuracy: 0,
              precision: 0,
              recall: 0,
              f1Score: 0,
              falsePositiveRate: 0,
            },
            baseline: {
              accuracy: 0,
              precision: 0,
              recall: 0,
              f1Score: 0,
              falsePositiveRate: 0,
            },
          },
        };
      }

      // Normalize metrics
      const recent = {
        accuracy: parseFloat(recentMetrics.avg_accuracy) || 0,
        precision: parseFloat(recentMetrics.avg_precision) || 0,
        recall: parseFloat(recentMetrics.avg_recall) || 0,
        f1Score: parseFloat(recentMetrics.avg_f1) || 0,
        falsePositiveRate: parseFloat(recentMetrics.avg_fpr) || 0,
      };

      const baseline = {
        accuracy: parseFloat(baselineMetrics.avg_accuracy) || 0,
        precision: parseFloat(baselineMetrics.avg_precision) || 0,
        recall: parseFloat(baselineMetrics.avg_recall) || 0,
        f1Score: parseFloat(baselineMetrics.avg_f1) || 0,
        falsePositiveRate: parseFloat(baselineMetrics.avg_fpr) || 0,
      };

      // Calculate drift scores
      const f1Drift = Math.abs(recent.f1Score - baseline.f1Score);
      const fprDrift = Math.abs(recent.falsePositiveRate - baseline.falsePositiveRate);
      const driftScore = (f1Drift + fprDrift) / 2;

      // Determine if drift is detected
      const hasDrift =
        driftScore > this.driftThreshold ||
        recent.f1Score < baseline.f1Score - 0.03 ||
        recent.falsePositiveRate > baseline.falsePositiveRate + 0.02;

      let recommendation: string | undefined;
      if (hasDrift) {
        if (recent.f1Score < baseline.f1Score - 0.03) {
          recommendation = 'Model performance degraded significantly. Consider retraining.';
        } else if (recent.falsePositiveRate > baseline.falsePositiveRate + 0.02) {
          recommendation = 'False positive rate increased. Review and retrain model.';
        } else {
          recommendation = 'Drift detected. Monitor closely and consider retraining.';
        }
      }

      return {
        hasDrift,
        driftScore,
        metrics: {
          recent,
          baseline,
        },
        recommendation,
      };
    } catch (error: any) {
      logger.error(`Failed to detect drift for model ${modelId}: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Check drift for all active models
   */
  async checkAllModels(windowDays: number = 7): Promise<Map<string, DriftResult>> {
    try {
      const modelRepository = this.dataSource.getRepository(MLModel);
      const activeModels = await modelRepository.find({
        where: { is_active: true },
      });

      const results = new Map<string, DriftResult>();

      for (const model of activeModels) {
        const driftResult = await this.detectDrift(model.id, windowDays);
        results.set(model.id, driftResult);

        if (driftResult.hasDrift) {
          logger.warn(`Drift detected for model ${model.model_type} v${model.version}: ${driftResult.recommendation}`);
        }
      }

      return results;
    } catch (error: any) {
      logger.error(`Failed to check drift for all models: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Check for data distribution drift (input data changes)
   */
  async checkDataDrift(modelType: string): Promise<boolean> {
    // This would compare feature distributions
    // Implementation depends on feature store backend
    // For now, return false (no drift detected)
    logger.info(`Data drift check for ${modelType} - not implemented yet`);
    return false;
  }
}
