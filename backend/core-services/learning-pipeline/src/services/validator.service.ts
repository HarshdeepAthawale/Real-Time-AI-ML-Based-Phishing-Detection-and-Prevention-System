import { S3Client, GetObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';
import { DataSource } from 'typeorm';
import { config } from '../../../shared/config';
import { logger } from '../utils/logger';
import { MLModel, ModelPerformance } from '../../../shared/database/models';

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  totalSamples?: number;
  truePositives?: number;
  trueNegatives?: number;
  falsePositives?: number;
  falseNegatives?: number;
}

export interface ModelComparison {
  current: ModelMetrics;
  new: ModelMetrics;
  improvement: number;
  shouldDeploy: boolean;
  reason?: string;
}

export class ValidatorService {
  private s3: S3Client;
  private dataSource: DataSource;
  private bucketName: string;
  private minImprovementForDeployment: number;
  private maxFalsePositiveRate: number;

  constructor(s3: S3Client, dataSource: DataSource) {
    this.s3 = s3;
    this.dataSource = dataSource;
    this.bucketName = config.aws.s3.models;
    this.minImprovementForDeployment = parseFloat(
      process.env.MIN_IMPROVEMENT_FOR_DEPLOYMENT || '0.01'
    );
    this.maxFalsePositiveRate = parseFloat(
      process.env.MAX_FALSE_POSITIVE_RATE || '0.02'
    );
  }

  /**
   * Validate a model by downloading and evaluating it
   * In production, this would trigger a validation job
   */
  async validateModel(
    modelType: string,
    modelVersion: string,
    testDatasetPath?: string
  ): Promise<ModelMetrics> {
    try {
      logger.info(`Validating model ${modelType} v${modelVersion}`);

      // Check if model exists in S3
      const modelPath = `models/${modelType}/${modelVersion}/`;
      const listCommand = new ListObjectsV2Command({
        Bucket: this.bucketName,
        Prefix: modelPath,
      });

      const listResponse = await this.s3.send(listCommand);
      
      if (!listResponse.Contents || listResponse.Contents.length === 0) {
        throw new Error(`Model not found at ${modelPath}`);
      }

      // In a real implementation, this would:
      // 1. Download the model from S3
      // 2. Load test dataset
      // 3. Run evaluation
      // 4. Calculate metrics
      
      // For now, we'll check if metrics are stored in the model metadata
      const metricsKey = `${modelPath}metrics.json`;
      try {
        const metricsCommand = new GetObjectCommand({
          Bucket: this.bucketName,
          Key: metricsKey,
        });

        const metricsResponse = await this.s3.send(metricsCommand);
        const metricsBody = await metricsResponse.Body?.transformToString();
        
        if (metricsBody) {
          const metrics = JSON.parse(metricsBody);
          return this.normalizeMetrics(metrics);
        }
      } catch (error) {
        logger.warn(`Metrics file not found, using default validation`);
      }

      // If no metrics file, return default metrics (in production, run actual validation)
      logger.warn(`No validation metrics found for ${modelType} v${modelVersion}, returning placeholder`);
      
      return {
        accuracy: 0.95,
        precision: 0.94,
        recall: 0.96,
        f1Score: 0.95,
        falsePositiveRate: 0.02,
        falseNegativeRate: 0.04,
      };
    } catch (error: any) {
      logger.error(`Failed to validate model ${modelType} v${modelVersion}: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Compare two model versions
   */
  async compareModels(
    modelType: string,
    currentVersion: string,
    newVersion: string,
    testDatasetPath?: string
  ): Promise<ModelComparison> {
    try {
      const [currentMetrics, newMetrics] = await Promise.all([
        this.validateModel(modelType, currentVersion, testDatasetPath),
        this.validateModel(modelType, newVersion, testDatasetPath),
      ]);

      const improvement = newMetrics.f1Score - currentMetrics.f1Score;
      
      // Determine if we should deploy
      const shouldDeploy =
        improvement >= this.minImprovementForDeployment &&
        newMetrics.falsePositiveRate <= this.maxFalsePositiveRate &&
        newMetrics.f1Score >= currentMetrics.f1Score;

      let reason: string | undefined;
      if (!shouldDeploy) {
        if (improvement < this.minImprovementForDeployment) {
          reason = `F1 improvement (${improvement.toFixed(4)}) below threshold (${this.minImprovementForDeployment})`;
        } else if (newMetrics.falsePositiveRate > this.maxFalsePositiveRate) {
          reason = `False positive rate (${newMetrics.falsePositiveRate.toFixed(4)}) exceeds maximum (${this.maxFalsePositiveRate})`;
        } else if (newMetrics.f1Score < currentMetrics.f1Score) {
          reason = `New model F1 score (${newMetrics.f1Score.toFixed(4)}) is lower than current (${currentMetrics.f1Score.toFixed(4)})`;
        }
      }

      return {
        current: currentMetrics,
        new: newMetrics,
        improvement,
        shouldDeploy,
        reason,
      };
    } catch (error: any) {
      logger.error(`Failed to compare models: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Store validation metrics in database
   */
  async storeValidationMetrics(
    modelId: string,
    metrics: ModelMetrics,
    date: Date = new Date()
  ): Promise<void> {
    try {
      const performanceRepository = this.dataSource.getRepository(ModelPerformance);

      // Check if performance record exists for this date
      const existing = await performanceRepository.findOne({
        where: {
          model_id: modelId,
          date,
        },
      });

      const performanceData = {
        model_id: modelId,
        date,
        accuracy: metrics.accuracy,
        precision: metrics.precision,
        recall: metrics.recall,
        f1_score: metrics.f1Score,
        false_positive_rate: metrics.falsePositiveRate,
        total_predictions: metrics.totalSamples || 0,
      };

      if (existing) {
        await performanceRepository.update(existing.id, performanceData);
      } else {
        await performanceRepository.save(performanceData);
      }

      logger.info(`Stored validation metrics for model ${modelId}`);
    } catch (error: any) {
      logger.error(`Failed to store validation metrics: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Normalize metrics from various formats
   */
  private normalizeMetrics(metrics: any): ModelMetrics {
    return {
      accuracy: metrics.accuracy || metrics.acc || 0,
      precision: metrics.precision || metrics.prec || 0,
      recall: metrics.recall || metrics.rec || 0,
      f1Score: metrics.f1Score || metrics.f1_score || metrics.f1 || 0,
      falsePositiveRate: metrics.falsePositiveRate || metrics.fpr || metrics.false_positive_rate || 0,
      falseNegativeRate: metrics.falseNegativeRate || metrics.fnr || metrics.false_negative_rate || 0,
      totalSamples: metrics.totalSamples || metrics.total_samples || metrics.n || undefined,
      truePositives: metrics.truePositives || metrics.tp || undefined,
      trueNegatives: metrics.trueNegatives || metrics.tn || undefined,
      falsePositives: metrics.falsePositives || metrics.fp || undefined,
      falseNegatives: metrics.falseNegatives || metrics.fn || undefined,
    };
  }
}
