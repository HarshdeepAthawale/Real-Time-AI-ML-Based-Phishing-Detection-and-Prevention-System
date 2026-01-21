import { S3Client, ListObjectsV2Command, GetObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3';
import { config } from '../../../shared/config';
import { logger } from '../utils/logger';

export interface DatasetInfo {
  path: string;
  version: string;
  size: number;
  createdAt: Date;
  statistics?: Record<string, any>;
}

export class FeatureStoreService {
  private s3: S3Client;
  private bucketName: string;

  constructor(s3: S3Client) {
    this.s3 = s3;
    this.bucketName = config.aws.s3.training;
  }

  /**
   * Get S3 path to training dataset for a model type and version
   */
  async getTrainingDataset(modelType: string, version: string): Promise<string> {
    const path = `datasets/${modelType}/${version}/`;
    return `s3://${this.bucketName}/${path}`;
  }

  /**
   * List available datasets for a model type
   */
  async listDatasets(modelType: string): Promise<DatasetInfo[]> {
    try {
      const prefix = `datasets/${modelType}/`;
      const command = new ListObjectsV2Command({
        Bucket: this.bucketName,
        Prefix: prefix,
        Delimiter: '/',
      });

      const response = await this.s3.send(command);
      const datasets: DatasetInfo[] = [];

      if (response.CommonPrefixes) {
        for (const prefix of response.CommonPrefixes) {
          const version = prefix.Prefix?.replace(`datasets/${modelType}/`, '').replace('/', '') || '';
          if (version) {
            datasets.push({
              path: `s3://${this.bucketName}/${prefix.Prefix}`,
              version,
              size: 0,
              createdAt: new Date(),
            });
          }
        }
      }

      return datasets.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
    } catch (error: any) {
      logger.error(`Failed to list datasets for ${modelType}: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Prepare dataset by aggregating collected data
   * This creates a new dataset version with all collected training data
   */
  async prepareDataset(modelType: string): Promise<string> {
    try {
      const timestamp = Date.now();
      const version = `v${timestamp}`;
      const datasetPath = `datasets/${modelType}/${version}/`;

      // List all collected data files
      const dataTypes = ['feedback', 'threats', 'false-positives'];
      const allData: any[] = [];

      for (const dataType of dataTypes) {
        const prefix = `training-data/${dataType}/`;
        const listCommand = new ListObjectsV2Command({
          Bucket: this.bucketName,
          Prefix: prefix,
        });

        const listResponse = await this.s3.send(listCommand);
        
        if (listResponse.Contents) {
          for (const object of listResponse.Contents) {
            if (object.Key && object.Key.endsWith('.json')) {
              const getCommand = new GetObjectCommand({
                Bucket: this.bucketName,
                Key: object.Key,
              });

              const getResponse = await this.s3.send(getCommand);
              const body = await getResponse.Body?.transformToString();
              
              if (body) {
                const data = JSON.parse(body);
                if (Array.isArray(data)) {
                  allData.push(...data);
                } else {
                  allData.push(data);
                }
              }
            }
          }
        }
      }

      if (allData.length === 0) {
        throw new Error(`No training data found for ${modelType}`);
      }

      // Save aggregated dataset
      const datasetKey = `${datasetPath}dataset.json`;
      await this.s3.send(
        new PutObjectCommand({
          Bucket: this.bucketName,
          Key: datasetKey,
          Body: JSON.stringify(allData, null, 2),
          ContentType: 'application/json',
        })
      );

      // Save dataset metadata
      const metadata = {
        modelType,
        version,
        size: allData.length,
        createdAt: new Date().toISOString(),
        dataTypes,
        statistics: {
          totalSamples: allData.length,
          feedbackSamples: allData.filter((d) => d.feedback_type).length,
          threatSamples: allData.filter((d) => d.threat_type).length,
          falsePositiveSamples: allData.filter((d) => !d.feedback_type && !d.threat_type).length,
        },
      };

      const metadataKey = `${datasetPath}metadata.json`;
      await this.s3.send(
        new PutObjectCommand({
          Bucket: this.bucketName,
          Key: metadataKey,
          Body: JSON.stringify(metadata, null, 2),
          ContentType: 'application/json',
        })
      );

      const s3Path = `s3://${this.bucketName}/${datasetPath}`;
      logger.info(`Prepared dataset for ${modelType}: ${s3Path} (${allData.length} samples)`);
      
      return s3Path;
    } catch (error: any) {
      logger.error(`Failed to prepare dataset for ${modelType}: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Get feature statistics for drift detection
   */
  async getFeatureStatistics(modelType: string, version?: string): Promise<Record<string, any>> {
    try {
      const datasetVersion = version || (await this.getLatestVersion(modelType));
      const metadataKey = `datasets/${modelType}/${datasetVersion}/metadata.json`;

      const command = new GetObjectCommand({
        Bucket: this.bucketName,
        Key: metadataKey,
      });

      const response = await this.s3.send(command);
      const body = await response.Body?.transformToString();
      
      if (body) {
        const metadata = JSON.parse(body);
        return metadata.statistics || {};
      }

      return {};
    } catch (error: any) {
      logger.warn(`Failed to get feature statistics: ${error.message}`);
      return {};
    }
  }

  /**
   * Get latest dataset version for a model type
   */
  private async getLatestVersion(modelType: string): Promise<string> {
    const datasets = await this.listDatasets(modelType);
    return datasets.length > 0 ? datasets[0].version : 'v0';
  }
}
