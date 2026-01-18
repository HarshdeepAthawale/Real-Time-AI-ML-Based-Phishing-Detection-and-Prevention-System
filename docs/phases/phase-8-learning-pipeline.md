# Phase 8: Continuous Learning & Model Training Pipeline

## Objective
Build an automated ML pipeline for incremental learning, model retraining, validation, A/B testing, and drift detection from user feedback and new threat data.

## Prerequisites
- Phases 1-7 completed
- ML models deployed (NLP, URL, Visual)
- Access to training infrastructure (GPU recommended)
- Python 3.11+ for training scripts

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         Learning Pipeline                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Data       │  │   Feature    │  │   Training   │  │
│  │  Collector   │→ │    Store     │→ │  Orchestrator│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Model      │  │   A/B        │  │   Drift      │  │
│  │  Validator   │  │   Testing    │  │   Detector   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
backend/core-services/learning-pipeline/
├── src/
│   ├── index.ts                    # Orchestration service
│   ├── services/
│   │   ├── data-collector.service.ts # Collect training data
│   │   ├── feature-store.service.ts   # Feature store management
│   │   ├── training-orchestrator.service.ts # Training coordination
│   │   ├── validator.service.ts       # Model validation
│   │   ├── drift-detector.service.ts  # Model drift detection
│   │   └── deployment.service.ts      # Model deployment
│   ├── jobs/
│   │   ├── scheduled-training.job.ts
│   │   └── drift-check.job.ts
│   └── utils/
│       └── metrics.ts
│
backend/ml-services/training/
├── data-pipeline/
│   ├── collect_feedback.py         # Collect user feedback
│   ├── prepare_dataset.py          # Dataset preparation
│   └── feature_extraction.py       # Feature extraction
├── nlp/
│   ├── train_phishing_model.py
│   └── train_ai_detector.py
├── url/
│   └── train_gnn_model.py
├── visual/
│   └── train_cnn_model.py
└── validation/
    ├── evaluate_model.py
    └── compare_models.py
```

## Implementation Steps

### 1. Data Collector Service

**File**: `backend/core-services/learning-pipeline/src/services/data-collector.service.ts`

```typescript
import { Pool } from 'pg';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import { logger } from '../utils/logger';

export class DataCollectorService {
  private pool: Pool;
  private s3: S3Client;
  private bucketName: string;
  
  constructor(pool: Pool, s3: S3Client, bucketName: string) {
    this.pool = pool;
    this.s3 = s3;
    this.bucketName = bucketName;
  }
  
  async collectFeedbackData(since: Date): Promise<number> {
    // Collect user feedback on detections
    const query = `
      SELECT 
        d.id,
        d.input_data,
        d.analysis_result,
        df.feedback_type,
        df.comment,
        d.detected_at
      FROM detections d
      JOIN detection_feedback df ON d.id = df.detection_id
      WHERE df.created_at >= $1
      ORDER BY df.created_at DESC
    `;
    
    const result = await this.pool.query(query, [since]);
    const feedbackData = result.rows;
    
    // Save to S3 for training
    const key = `training-data/feedback/${Date.now()}.json`;
    await this.s3.send(new PutObjectCommand({
      Bucket: this.bucketName,
      Key: key,
      Body: JSON.stringify(feedbackData),
      ContentType: 'application/json'
    }));
    
    logger.info(`Collected ${feedbackData.length} feedback records`);
    return feedbackData.length;
  }
  
  async collectThreatData(since: Date): Promise<number> {
    // Collect confirmed threats for training
    const query = `
      SELECT 
        t.id,
        t.threat_type,
        t.source,
        t.source_value,
        t.metadata,
        d.analysis_result
      FROM threats t
      JOIN detections d ON t.id = d.threat_id
      WHERE t.status IN ('blocked', 'resolved')
        AND t.detected_at >= $1
    `;
    
    const result = await this.pool.query(query, [since]);
    const threatData = result.rows;
    
    const key = `training-data/threats/${Date.now()}.json`;
    await this.s3.send(new PutObjectCommand({
      Bucket: this.bucketName,
      Key: key,
      Body: JSON.stringify(threatData),
      ContentType: 'application/json'
    }));
    
    logger.info(`Collected ${threatData.length} threat records`);
    return threatData.length;
  }
  
  async collectFalsePositives(since: Date): Promise<number> {
    // Collect false positives for training
    const query = `
      SELECT 
        d.id,
        d.input_data,
        d.analysis_result
      FROM detections d
      JOIN detection_feedback df ON d.id = df.detection_id
      WHERE df.feedback_type = 'false_positive'
        AND df.created_at >= $1
    `;
    
    const result = await this.pool.query(query, [since]);
    const falsePositives = result.rows;
    
    const key = `training-data/false-positives/${Date.now()}.json`;
    await this.s3.send(new PutObjectCommand({
      Bucket: this.bucketName,
      Key: key,
      Body: JSON.stringify(falsePositives),
      ContentType: 'application/json'
    }));
    
    logger.info(`Collected ${falsePositives.length} false positive records`);
    return falsePositives.length;
  }
}
```

### 2. Feature Store Service

**File**: `backend/core-services/learning-pipeline/src/services/feature-store.service.ts`

```typescript
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import { logger } from '../utils/logger';

export class FeatureStoreService {
  private s3: S3Client;
  private bucketName: string;
  
  constructor(s3: S3Client, bucketName: string) {
    this.s3 = s3;
    this.bucketName = bucketName;
  }
  
  async getTrainingDataset(modelType: string, version: string): Promise<string> {
    // Return S3 path to training dataset
    return `s3://${this.bucketName}/datasets/${modelType}/${version}/`;
  }
  
  async prepareDataset(modelType: string): Promise<string> {
    // Aggregate and prepare dataset for training
    const datasetPath = `datasets/${modelType}/${Date.now()}/`;
    
    // This would call Python script to prepare dataset
    // For now, return the path
    return `s3://${this.bucketName}/${datasetPath}`;
  }
  
  async getFeatureStatistics(modelType: string): Promise<any> {
    // Get feature statistics for drift detection
    // Implementation depends on feature store backend
    return {};
  }
}
```

### 3. Training Orchestrator Service

**File**: `backend/core-services/learning-pipeline/src/services/training-orchestrator.service.ts`

```typescript
import { ECSClient, RunTaskCommand } from '@aws-sdk/client-ecs';
import { logger } from '../utils/logger';

export class TrainingOrchestratorService {
  private ecs: ECSClient;
  private clusterName: string;
  private taskDefinition: string;
  
  constructor(ecs: ECSClient, clusterName: string, taskDefinition: string) {
    this.ecs = ecs;
    this.clusterName = clusterName;
    this.taskDefinition = taskDefinition;
  }
  
  async triggerTraining(modelType: string, datasetPath: string): Promise<string> {
    // Trigger training job on ECS/EKS
    const command = this.getTrainingCommand(modelType, datasetPath);
    
    const task = await this.ecs.send(new RunTaskCommand({
      cluster: this.clusterName,
      taskDefinition: this.taskDefinition,
      launchType: 'FARGATE',
      networkConfiguration: {
        awsvpcConfiguration: {
          subnets: ['subnet-xxx'], // Your subnet IDs
          assignPublicIp: 'ENABLED'
        }
      },
      overrides: {
        containerOverrides: [{
          name: 'training-container',
          command: command.split(' ')
        }]
      }
    }));
    
    const taskId = task.tasks?.[0]?.taskArn || 'unknown';
    logger.info(`Training task started: ${taskId} for model ${modelType}`);
    
    return taskId;
  }
  
  private getTrainingCommand(modelType: string, datasetPath: string): string {
    const commands: Record<string, string> = {
      'nlp': `python training/nlp/train_phishing_model.py --dataset ${datasetPath}`,
      'url': `python training/url/train_gnn_model.py --dataset ${datasetPath}`,
      'visual': `python training/visual/train_cnn_model.py --dataset ${datasetPath}`
    };
    
    return commands[modelType] || '';
  }
}
```

### 4. Model Validator Service

**File**: `backend/core-services/learning-pipeline/src/services/validator.service.ts`

```typescript
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import { logger } from '../utils/logger';

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
}

export class ValidatorService {
  private s3: S3Client;
  private bucketName: string;
  
  constructor(s3: S3Client, bucketName: string) {
    this.s3 = s3;
    this.bucketName = bucketName;
  }
  
  async validateModel(
    modelType: string,
    modelVersion: string,
    testDatasetPath: string
  ): Promise<ModelMetrics> {
    // Trigger validation job
    // This would call Python validation script
    // For now, return mock metrics
    
    logger.info(`Validating model ${modelType} v${modelVersion}`);
    
    // In real implementation, this would:
    // 1. Download model from S3
    // 2. Load test dataset
    // 3. Run evaluation
    // 4. Return metrics
    
    return {
      accuracy: 0.95,
      precision: 0.94,
      recall: 0.96,
      f1Score: 0.95,
      falsePositiveRate: 0.02,
      falseNegativeRate: 0.04
    };
  }
  
  async compareModels(
    currentModel: string,
    newModel: string,
    testDatasetPath: string
  ): Promise<{
    current: ModelMetrics;
    new: ModelMetrics;
    improvement: number;
    shouldDeploy: boolean;
  }> {
    const currentMetrics = await this.validateModel('nlp', currentModel, testDatasetPath);
    const newMetrics = await this.validateModel('nlp', newModel, testDatasetPath);
    
    const improvement = newMetrics.f1Score - currentMetrics.f1Score;
    const shouldDeploy = improvement > 0.01 && newMetrics.falsePositiveRate < 0.02;
    
    return {
      current: currentMetrics,
      new: newMetrics,
      improvement,
      shouldDeploy
    };
  }
}
```

### 5. Drift Detector Service

**File**: `backend/core-services/learning-pipeline/src/services/drift-detector.service.ts`

```typescript
import { Pool } from 'pg';
import { logger } from '../utils/logger';

export class DriftDetectorService {
  private pool: Pool;
  
  constructor(pool: Pool) {
    this.pool = pool;
  }
  
  async detectDrift(modelId: string, windowDays: number = 7): Promise<{
    hasDrift: boolean;
    driftScore: number;
    metrics: any;
  }> {
    // Compare recent performance with historical baseline
    const query = `
      SELECT 
        AVG(accuracy) as avg_accuracy,
        AVG(precision) as avg_precision,
        AVG(recall) as avg_recall,
        AVG(f1_score) as avg_f1,
        AVG(false_positive_rate) as avg_fpr
      FROM model_performance
      WHERE model_id = $1
        AND date >= CURRENT_DATE - INTERVAL '${windowDays} days'
    `;
    
    const recent = await this.pool.query(query, [modelId]);
    
    const baselineQuery = `
      SELECT 
        AVG(accuracy) as avg_accuracy,
        AVG(precision) as avg_precision,
        AVG(recall) as avg_recall,
        AVG(f1_score) as avg_f1,
        AVG(false_positive_rate) as avg_fpr
      FROM model_performance
      WHERE model_id = $1
        AND date < CURRENT_DATE - INTERVAL '${windowDays} days'
        AND date >= CURRENT_DATE - INTERVAL '${windowDays * 2} days'
    `;
    
    const baseline = await this.pool.query(baselineQuery, [modelId]);
    
    if (recent.rows.length === 0 || baseline.rows.length === 0) {
      return {
        hasDrift: false,
        driftScore: 0,
        metrics: {}
      };
    }
    
    const recentMetrics = recent.rows[0];
    const baselineMetrics = baseline.rows[0];
    
    // Calculate drift score
    const f1Drift = Math.abs(recentMetrics.avg_f1 - baselineMetrics.avg_f1);
    const fprDrift = Math.abs(recentMetrics.avg_fpr - baselineMetrics.avg_fpr);
    const driftScore = (f1Drift + fprDrift) / 2;
    
    const hasDrift = driftScore > 0.05 || recentMetrics.avg_f1 < baselineMetrics.avg_f1 - 0.03;
    
    return {
      hasDrift,
      driftScore,
      metrics: {
        recent: recentMetrics,
        baseline: baselineMetrics
      }
    };
  }
  
  async checkDataDrift(modelType: string): Promise<boolean> {
    // Check if input data distribution has changed
    // This would compare feature distributions
    // Implementation depends on feature store
    return false;
  }
}
```

### 6. Model Deployment Service

**File**: `backend/core-services/learning-pipeline/src/services/deployment.service.ts`

```typescript
import { S3Client, CopyObjectCommand } from '@aws-sdk/client-s3';
import { ECSClient, UpdateServiceCommand } from '@aws-sdk/client-ecs';
import { logger } from '../utils/logger';

export class DeploymentService {
  private s3: S3Client;
  private ecs: ECSClient;
  private bucketName: string;
  
  constructor(s3: S3Client, ecs: ECSClient, bucketName: string) {
    this.s3 = s3;
    this.ecs = ecs;
    this.bucketName = bucketName;
  }
  
  async deployModel(
    modelType: string,
    modelVersion: string,
    deploymentStrategy: 'canary' | 'blue-green' | 'rolling' = 'canary'
  ): Promise<void> {
    logger.info(`Deploying ${modelType} model v${modelVersion} using ${deploymentStrategy} strategy`);
    
    // Copy model to production location
    const sourceKey = `models/${modelType}/${modelVersion}/model.bin`;
    const destKey = `models/${modelType}/production/model.bin`;
    
    await this.s3.send(new CopyObjectCommand({
      Bucket: this.bucketName,
      CopySource: `${this.bucketName}/${sourceKey}`,
      Key: destKey
    }));
    
    // Update model metadata
    await this.updateModelMetadata(modelType, modelVersion);
    
    // Trigger service restart to load new model
    if (deploymentStrategy === 'canary') {
      await this.deployCanary(modelType, modelVersion);
    } else {
      await this.restartService(modelType);
    }
  }
  
  private async deployCanary(modelType: string, version: string): Promise<void> {
    // Deploy to canary environment first
    // Route small percentage of traffic
    logger.info(`Deploying canary for ${modelType} v${version}`);
  }
  
  private async restartService(modelType: string): Promise<void> {
    // Force ECS service restart to load new model
    const serviceName = `${modelType}-service`;
    
    await this.ecs.send(new UpdateServiceCommand({
      cluster: 'your-cluster',
      service: serviceName,
      forceNewDeployment: true
    }));
  }
  
  private async updateModelMetadata(modelType: string, version: string): Promise<void> {
    // Update database with new model version
    // Mark as active
  }
}
```

### 7. Scheduled Training Job

**File**: `backend/core-services/learning-pipeline/src/jobs/scheduled-training.job.ts`

```typescript
import cron from 'node-cron';
import { DataCollectorService } from '../services/data-collector.service';
import { TrainingOrchestratorService } from '../services/training-orchestrator.service';
import { ValidatorService } from '../services/validator.service';
import { DeploymentService } from '../services/deployment.service';
import { logger } from '../utils/logger';

export class ScheduledTrainingJob {
  private dataCollector: DataCollectorService;
  private trainingOrchestrator: TrainingOrchestratorService;
  private validator: ValidatorService;
  private deployment: DeploymentService;
  
  constructor(
    dataCollector: DataCollectorService,
    trainingOrchestrator: TrainingOrchestratorService,
    validator: ValidatorService,
    deployment: DeploymentService
  ) {
    this.dataCollector = dataCollector;
    this.trainingOrchestrator = trainingOrchestrator;
    this.validator = validator;
    this.deployment = deployment;
  }
  
  start(): void {
    // Run weekly training
    cron.schedule('0 2 * * 0', async () => { // Every Sunday at 2 AM
      await this.runTrainingPipeline('nlp');
      await this.runTrainingPipeline('url');
      await this.runTrainingPipeline('visual');
    });
    
    // Run daily data collection
    cron.schedule('0 1 * * *', async () => { // Every day at 1 AM
      const since = new Date();
      since.setDate(since.getDate() - 1);
      
      await this.dataCollector.collectFeedbackData(since);
      await this.dataCollector.collectThreatData(since);
      await this.dataCollector.collectFalsePositives(since);
    });
  }
  
  private async runTrainingPipeline(modelType: string): Promise<void> {
    try {
      logger.info(`Starting training pipeline for ${modelType}`);
      
      // 1. Prepare dataset
      const datasetPath = await this.prepareDataset(modelType);
      
      // 2. Trigger training
      const taskId = await this.trainingOrchestrator.triggerTraining(modelType, datasetPath);
      logger.info(`Training task started: ${taskId}`);
      
      // 3. Wait for training to complete (polling or event-driven)
      // 4. Validate model
      // 5. Compare with current model
      // 6. Deploy if better
      
    } catch (error) {
      logger.error(`Training pipeline failed for ${modelType}`, error);
    }
  }
  
  private async prepareDataset(modelType: string): Promise<string> {
    // Implementation
    return '';
  }
}
```

### 8. Python Training Scripts

**File**: `backend/ml-services/training/nlp/train_phishing_model.py`

```python
import argparse
import json
import boto3
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch

def load_dataset(s3_path: str):
    """Load training dataset from S3"""
    s3 = boto3.client('s3')
    # Parse S3 path and download
    # Load JSON data
    # Convert to HuggingFace Dataset
    pass

def train_model(dataset_path: str, output_path: str):
    """Train phishing detection model"""
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Load base model
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_path}/logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_path)
    
    # Evaluate
    metrics = trainer.evaluate()
    print(f"Model metrics: {metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    train_model(args.dataset, args.output)
```

## Deliverables Checklist

- [ ] Data collector service implemented
- [ ] Feature store service implemented
- [ ] Training orchestrator implemented
- [ ] Model validator implemented
- [ ] Drift detector implemented
- [ ] Deployment service implemented
- [ ] Scheduled jobs configured
- [ ] Python training scripts created
- [ ] MLflow integration (optional)
- [ ] Tests written

## Next Steps

After completing Phase 8:
1. Set up training infrastructure
2. Prepare initial training datasets
3. Run initial model training
4. Configure monitoring and alerts
5. Proceed to Phase 9: Browser Extension Backend
