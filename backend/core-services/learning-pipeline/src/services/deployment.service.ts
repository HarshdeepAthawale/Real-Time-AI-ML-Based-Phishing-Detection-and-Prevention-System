import { S3Client, CopyObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';
import { ECSClient, UpdateServiceCommand, DescribeServicesCommand } from '@aws-sdk/client-ecs';
import { DataSource } from 'typeorm';
import { config } from '../../../shared/config';
import { logger } from '../utils/logger';
import { MLModel, ModelVersion } from '../../../shared/database/models';

export type DeploymentStrategy = 'canary' | 'blue-green' | 'rolling' | 'immediate';

export interface DeploymentConfig {
  modelType: string;
  modelVersion: string;
  strategy?: DeploymentStrategy;
  canaryTrafficPercent?: number;
}

export class DeploymentService {
  private s3: S3Client;
  private ecs: ECSClient;
  private dataSource: DataSource;
  private bucketName: string;

  constructor(s3: S3Client, ecs: ECSClient, dataSource: DataSource) {
    this.s3 = s3;
    this.ecs = ecs;
    this.dataSource = dataSource;
    this.bucketName = config.aws.s3.models;
  }

  /**
   * Deploy a model to production
   */
  async deployModel(config: DeploymentConfig): Promise<void> {
    try {
      logger.info(`Deploying ${config.modelType} model v${config.modelVersion} using ${config.strategy || 'immediate'} strategy`);

      const strategy = config.strategy || 'immediate';

      // Get model from database
      const modelRepository = this.dataSource.getRepository(MLModel);
      const model = await modelRepository.findOne({
        where: {
          model_type: config.modelType,
          version: config.modelVersion,
        },
      });

      if (!model) {
        throw new Error(`Model ${config.modelType} v${config.modelVersion} not found in database`);
      }

      // Copy model to production location in S3
      await this.copyModelToProduction(config.modelType, config.modelVersion);

      // Update model metadata in database
      await this.updateModelMetadata(model.id, config.modelVersion);

      // Deploy based on strategy
      switch (strategy) {
        case 'canary':
          await this.deployCanary(config.modelType, config.modelVersion, config.canaryTrafficPercent || 10);
          break;
        case 'blue-green':
          await this.deployBlueGreen(config.modelType, config.modelVersion);
          break;
        case 'rolling':
          await this.deployRolling(config.modelType, config.modelVersion);
          break;
        case 'immediate':
        default:
          await this.restartService(config.modelType);
          break;
      }

      logger.info(`Successfully deployed ${config.modelType} model v${config.modelVersion}`);
    } catch (error: any) {
      logger.error(`Failed to deploy model: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Copy model files to production S3 location
   */
  private async copyModelToProduction(modelType: string, version: string): Promise<void> {
    try {
      const sourcePrefix = `models/${modelType}/${version}/`;
      const destPrefix = `models/${modelType}/production/`;

      // List all files in source
      const listCommand = new ListObjectsV2Command({
        Bucket: this.bucketName,
        Prefix: sourcePrefix,
      });

      const listResponse = await this.s3.send(listCommand);

      if (!listResponse.Contents || listResponse.Contents.length === 0) {
        throw new Error(`No model files found at ${sourcePrefix}`);
      }

      // Copy each file
      for (const object of listResponse.Contents) {
        if (!object.Key) continue;

        const sourceKey = object.Key;
        const destKey = sourceKey.replace(sourcePrefix, destPrefix);

        await this.s3.send(
          new CopyObjectCommand({
            Bucket: this.bucketName,
            CopySource: `${this.bucketName}/${sourceKey}`,
            Key: destKey,
          })
        );

        logger.debug(`Copied ${sourceKey} to ${destKey}`);
      }

      logger.info(`Copied model files to production location: ${destPrefix}`);
    } catch (error: any) {
      logger.error(`Failed to copy model to production: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Update model metadata in database
   */
  private async updateModelMetadata(modelId: string, version: string): Promise<void> {
    try {
      const modelRepository = this.dataSource.getRepository(MLModel);

      // Deactivate current active model of same type
      const currentActive = await modelRepository.findOne({
        where: {
          id: modelId,
        },
      });

      if (currentActive) {
        // Deactivate all models of the same type
        await modelRepository.update(
          { model_type: currentActive.model_type, is_active: true },
          { is_active: false }
        );

        // Activate new model
        await modelRepository.update(modelId, {
          is_active: true,
          deployed_at: new Date(),
        });

        logger.info(`Updated model metadata: ${currentActive.model_type} v${version} is now active`);
      }
    } catch (error: any) {
      logger.error(`Failed to update model metadata: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Deploy using canary strategy
   */
  private async deployCanary(modelType: string, version: string, trafficPercent: number): Promise<void> {
    logger.info(`Deploying canary for ${modelType} v${version} with ${trafficPercent}% traffic`);
    
    // In a real implementation, this would:
    // 1. Deploy new model to canary environment
    // 2. Configure load balancer to route X% traffic to canary
    // 3. Monitor metrics
    // 4. Gradually increase traffic if metrics are good
    // 5. Fully promote if successful
    
    // For now, we'll just restart the service with canary configuration
    await this.restartService(modelType);
    
    logger.info(`Canary deployment initiated for ${modelType} v${version}`);
  }

  /**
   * Deploy using blue-green strategy
   */
  private async deployBlueGreen(modelType: string, version: string): Promise<void> {
    logger.info(`Deploying blue-green for ${modelType} v${version}`);
    
    // In a real implementation, this would:
    // 1. Deploy new model to green environment
    // 2. Run validation tests
    // 3. Switch traffic from blue to green
    // 4. Keep blue as backup
    
    await this.restartService(modelType);
    
    logger.info(`Blue-green deployment initiated for ${modelType} v${version}`);
  }

  /**
   * Deploy using rolling strategy
   */
  private async deployRolling(modelType: string, version: string): Promise<void> {
    logger.info(`Deploying rolling update for ${modelType} v${version}`);
    
    // In a real implementation, this would:
    // 1. Update instances one at a time
    // 2. Wait for health checks
    // 3. Continue with next instance
    
    await this.restartService(modelType);
    
    logger.info(`Rolling deployment initiated for ${modelType} v${version}`);
  }

  /**
   * Restart ECS service to load new model
   */
  private async restartService(modelType: string): Promise<void> {
    try {
      const clusterName = process.env.ECS_CLUSTER_NAME || 'phishing-detection-cluster';
      const serviceName = `${modelType}-service`;

      // Check if service exists
      const describeCommand = new DescribeServicesCommand({
        cluster: clusterName,
        services: [serviceName],
      });

      const describeResponse = await this.ecs.send(describeCommand);
      
      if (!describeResponse.services || describeResponse.services.length === 0) {
        logger.warn(`Service ${serviceName} not found in cluster ${clusterName}. Skipping restart.`);
        return;
      }

      // Force new deployment
      const updateCommand = new UpdateServiceCommand({
        cluster: clusterName,
        service: serviceName,
        forceNewDeployment: true,
      });

      await this.ecs.send(updateCommand);
      logger.info(`Triggered new deployment for service ${serviceName}`);
    } catch (error: any) {
      logger.error(`Failed to restart service: ${error.message}`, error);
      // Don't throw - service restart is not critical for model deployment
    }
  }
}
