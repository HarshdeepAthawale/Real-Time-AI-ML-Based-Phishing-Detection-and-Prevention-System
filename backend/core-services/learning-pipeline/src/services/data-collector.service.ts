import { DataSource } from 'typeorm';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { config } from '../../../shared/config';
import { logger } from '../utils/logger';
import { Detection, DetectionFeedback, Threat } from '../../../shared/database/models';

export class DataCollectorService {
  private dataSource: DataSource;
  private s3: S3Client;
  private bucketName: string;

  constructor(dataSource: DataSource, s3: S3Client) {
    this.dataSource = dataSource;
    this.s3 = s3;
    this.bucketName = config.aws.s3.training;
  }

  /**
   * Collect user feedback data from detections
   */
  async collectFeedbackData(since: Date): Promise<number> {
    try {
      const detectionRepository = this.dataSource.getRepository(Detection);
      const feedbackRepository = this.dataSource.getRepository(DetectionFeedback);

      const query = detectionRepository
        .createQueryBuilder('detection')
        .leftJoinAndSelect('detection.feedback', 'feedback')
        .where('feedback.created_at >= :since', { since })
        .orderBy('feedback.created_at', 'DESC');

      const detections = await query.getMany();

      const feedbackData = detections
        .filter((d) => d.feedback && d.feedback.length > 0)
        .map((detection) => {
          const latestFeedback = detection.feedback[0];
          return {
            detection_id: detection.id,
            input_data: detection.input_data,
            analysis_result: detection.analysis_result,
            feedback_type: latestFeedback.feedback_type,
            comment: latestFeedback.comment,
            detected_at: detection.detected_at,
            created_at: latestFeedback.created_at,
          };
        });

      if (feedbackData.length === 0) {
        logger.info('No feedback data collected');
        return 0;
      }

      // Save to S3
      const timestamp = Date.now();
      const key = `training-data/feedback/${timestamp}.json`;
      
      await this.s3.send(
        new PutObjectCommand({
          Bucket: this.bucketName,
          Key: key,
          Body: JSON.stringify(feedbackData, null, 2),
          ContentType: 'application/json',
        })
      );

      logger.info(`Collected ${feedbackData.length} feedback records, saved to s3://${this.bucketName}/${key}`);
      return feedbackData.length;
    } catch (error: any) {
      logger.error(`Failed to collect feedback data: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Collect confirmed threat data for training
   */
  async collectThreatData(since: Date): Promise<number> {
    try {
      const threatRepository = this.dataSource.getRepository(Threat);
      const detectionRepository = this.dataSource.getRepository(Detection);

      const threats = await threatRepository
        .createQueryBuilder('threat')
        .leftJoinAndSelect('threat.detections', 'detection')
        .where('threat.status IN (:...statuses)', { statuses: ['blocked', 'resolved'] })
        .andWhere('threat.detected_at >= :since', { since })
        .orderBy('threat.detected_at', 'DESC')
        .getMany();

      const threatData = threats.map((threat) => {
        const detection = threat.detections?.[0];
        return {
          threat_id: threat.id,
          threat_type: threat.threat_type,
          source: threat.source,
          source_value: threat.source_value,
          metadata: threat.metadata,
          severity: threat.severity,
          confidence_score: threat.confidence_score,
          analysis_result: detection?.analysis_result || {},
          detected_at: threat.detected_at,
        };
      });

      if (threatData.length === 0) {
        logger.info('No threat data collected');
        return 0;
      }

      // Save to S3
      const timestamp = Date.now();
      const key = `training-data/threats/${timestamp}.json`;

      await this.s3.send(
        new PutObjectCommand({
          Bucket: this.bucketName,
          Key: key,
          Body: JSON.stringify(threatData, null, 2),
          ContentType: 'application/json',
        })
      );

      logger.info(`Collected ${threatData.length} threat records, saved to s3://${this.bucketName}/${key}`);
      return threatData.length;
    } catch (error: any) {
      logger.error(`Failed to collect threat data: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Collect false positive detections for training
   */
  async collectFalsePositives(since: Date): Promise<number> {
    try {
      const detectionRepository = this.dataSource.getRepository(Detection);
      const feedbackRepository = this.dataSource.getRepository(DetectionFeedback);

      const query = detectionRepository
        .createQueryBuilder('detection')
        .leftJoinAndSelect('detection.feedback', 'feedback')
        .where('feedback.feedback_type = :type', { type: 'false_positive' })
        .andWhere('feedback.created_at >= :since', { since })
        .orderBy('feedback.created_at', 'DESC');

      const detections = await query.getMany();

      const falsePositives = detections
        .filter((d) => d.feedback && d.feedback.length > 0)
        .map((detection) => ({
          detection_id: detection.id,
          input_data: detection.input_data,
          analysis_result: detection.analysis_result,
          confidence_score: detection.confidence_score,
          detected_at: detection.detected_at,
        }));

      if (falsePositives.length === 0) {
        logger.info('No false positive data collected');
        return 0;
      }

      // Save to S3
      const timestamp = Date.now();
      const key = `training-data/false-positives/${timestamp}.json`;

      await this.s3.send(
        new PutObjectCommand({
          Bucket: this.bucketName,
          Key: key,
          Body: JSON.stringify(falsePositives, null, 2),
          ContentType: 'application/json',
        })
      );

      logger.info(`Collected ${falsePositives.length} false positive records, saved to s3://${this.bucketName}/${key}`);
      return falsePositives.length;
    } catch (error: any) {
      logger.error(`Failed to collect false positives: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Collect all training data types
   */
  async collectAllTrainingData(since: Date): Promise<{
    feedback: number;
    threats: number;
    falsePositives: number;
  }> {
    const [feedback, threats, falsePositives] = await Promise.all([
      this.collectFeedbackData(since),
      this.collectThreatData(since),
      this.collectFalsePositives(since),
    ]);

    return {
      feedback,
      threats,
      falsePositives,
    };
  }
}
