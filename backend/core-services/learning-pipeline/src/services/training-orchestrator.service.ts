import { ECSClient, RunTaskCommand, DescribeTasksCommand } from '@aws-sdk/client-ecs';
import { DataSource } from 'typeorm';
import { config } from '../../../../shared/config';
import { logger } from '../utils/logger';
import { TrainingJob } from '../../../../shared/database/models';

export interface TrainingTaskConfig {
  modelType: string;
  datasetPath: string;
  trainingConfig?: Record<string, any>;
}

export class TrainingOrchestratorService {
  private ecs: ECSClient;
  private dataSource: DataSource;
  private clusterName: string;
  private taskDefinition: string;
  private subnetIds: string[];
  private securityGroupIds: string[];

  constructor(ecs: ECSClient, dataSource: DataSource) {
    this.ecs = ecs;
    this.dataSource = dataSource;
    this.clusterName = process.env.ECS_CLUSTER_NAME || 'phishing-detection-cluster';
    this.taskDefinition = process.env.ECS_TASK_DEFINITION || 'phishing-detection-training';
    
    const subnetIdsEnv = process.env.ECS_SUBNET_IDS || '';
    this.subnetIds = subnetIdsEnv.split(',').filter((id) => id.trim().length > 0);
    
    const securityGroupIdsEnv = process.env.ECS_SECURITY_GROUP_IDS || '';
    this.securityGroupIds = securityGroupIdsEnv.split(',').filter((id) => id.trim().length > 0);

    if (this.subnetIds.length === 0) {
      logger.warn('No ECS subnet IDs configured. Training tasks may fail.');
    }
  }

  /**
   * Trigger training job for a model type
   */
  async triggerTraining(config: TrainingTaskConfig): Promise<string> {
    try {
      const trainingJobRepository = this.dataSource.getRepository(TrainingJob);

      // Create training job record
      const trainingJob = trainingJobRepository.create({
        model_type: config.modelType,
        status: 'pending',
        training_config: config.trainingConfig || {},
        dataset_path_s3: config.datasetPath,
        started_at: new Date(),
      });

      const savedJob = await trainingJobRepository.save(trainingJob);

      // Prepare training command
      const command = this.getTrainingCommand(config.modelType, config.datasetPath);

      // Prepare environment variables
      const environment = [
        { name: 'MODEL_TYPE', value: config.modelType },
        { name: 'DATASET_PATH', value: config.datasetPath },
        { name: 'TRAINING_JOB_ID', value: savedJob.id },
        { name: 'S3_BUCKET_MODELS', value: process.env.S3_BUCKET_MODELS || 'phishing-detection-models-dev' },
        { name: 'S3_BUCKET_TRAINING', value: process.env.S3_BUCKET_TRAINING || 'phishing-detection-training-dev' },
        { name: 'AWS_REGION', value: process.env.AWS_REGION || 'us-east-1' },
      ];

      // Add training config as JSON
      if (config.trainingConfig) {
        environment.push({
          name: 'TRAINING_CONFIG',
          value: JSON.stringify(config.trainingConfig),
        });
      }

      // Run ECS task
      const runTaskCommand = new RunTaskCommand({
        cluster: this.clusterName,
        taskDefinition: this.taskDefinition,
        launchType: 'FARGATE',
        networkConfiguration: {
          awsvpcConfiguration: {
            subnets: this.subnetIds.length > 0 ? this.subnetIds : undefined,
            securityGroups: this.securityGroupIds.length > 0 ? this.securityGroupIds : undefined,
            assignPublicIp: 'ENABLED',
          },
        },
        overrides: {
          containerOverrides: [
            {
              name: 'training-container',
              command: command.split(' '),
              environment,
            },
          ],
        },
        tags: [
          { key: 'ModelType', value: config.modelType },
          { key: 'TrainingJobId', value: savedJob.id },
          { key: 'DatasetPath', value: config.datasetPath },
        ],
      });

      const taskResponse = await this.ecs.send(runTaskCommand);
      const taskArn = taskResponse.tasks?.[0]?.taskArn;

      if (!taskArn) {
        throw new Error('Failed to start training task - no task ARN returned');
      }

      // Update training job with task ARN
      savedJob.status = 'running';
      await trainingJobRepository.save(savedJob);

      logger.info(`Training task started: ${taskArn} for model ${config.modelType} (job: ${savedJob.id})`);
      
      return savedJob.id;
    } catch (error: any) {
      logger.error(`Failed to trigger training for ${config.modelType}: ${error.message}`, error);
      throw error;
    }
  }

  /**
   * Get training command for model type
   */
  private getTrainingCommand(modelType: string, datasetPath: string): string {
    const commands: Record<string, string> = {
      nlp: `python -m training.nlp.train_phishing_model --dataset "${datasetPath}"`,
      url: `python -m training.url.train_gnn_model --dataset "${datasetPath}"`,
      visual: `python -m training.visual.train_cnn_model --dataset "${datasetPath}"`,
    };

    const command = commands[modelType];
    if (!command) {
      throw new Error(`Unknown model type: ${modelType}`);
    }

    return command;
  }

  /**
   * Check training task status
   */
  async checkTaskStatus(taskArn: string): Promise<{
    status: string;
    stoppedReason?: string;
  }> {
    try {
      const command = new DescribeTasksCommand({
        cluster: this.clusterName,
        tasks: [taskArn],
      });

      const response = await this.ecs.send(command);
      const task = response.tasks?.[0];

      if (!task) {
        return { status: 'UNKNOWN' };
      }

      return {
        status: task.lastStatus || 'UNKNOWN',
        stoppedReason: task.stoppedReason,
      };
    } catch (error: any) {
      logger.error(`Failed to check task status: ${error.message}`, error);
      return { status: 'ERROR' };
    }
  }
}
