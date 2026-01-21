# Learning Pipeline Service

Automated ML pipeline for continuous learning, model retraining, validation, A/B testing, and drift detection.

## Overview

The Learning Pipeline service orchestrates the complete ML lifecycle:
1. **Data Collection** - Collects user feedback, threats, and false positives from the database
2. **Feature Store** - Manages training datasets in S3
3. **Training Orchestration** - Triggers ECS tasks to train models
4. **Model Validation** - Validates trained models and compares with production
5. **Drift Detection** - Monitors model performance for degradation
6. **Model Deployment** - Deploys validated models to production

## Architecture

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

## Services

### Data Collector Service
- Collects user feedback from `detection_feedback` table
- Collects confirmed threats from `threats` table
- Collects false positives for training
- Uploads data to S3 training bucket

### Feature Store Service
- Manages training datasets in S3
- Prepares datasets by aggregating collected data
- Tracks dataset versions and statistics

### Training Orchestrator Service
- Triggers ECS Fargate tasks for training
- Supports NLP, URL, and Visual model training
- Tracks training job status in database

### Validator Service
- Validates models on test datasets
- Compares new models with production
- Returns deployment recommendations

### Drift Detector Service
- Monitors model performance over time
- Detects performance degradation
- Alerts when drift threshold exceeded

### Deployment Service
- Deploys validated models to production
- Supports canary, blue-green, rolling strategies
- Updates ECS services to load new models

## Scheduled Jobs

### Data Collection
- **Schedule**: Daily at 1 AM (configurable via `DATA_COLLECTION_SCHEDULE`)
- **Action**: Collects feedback, threats, and false positives from last 24 hours

### Model Training
- **Schedule**: Weekly on Sunday at 2 AM (configurable via `TRAINING_SCHEDULE`)
- **Action**: Trains all model types (NLP, URL, Visual)

### Drift Check
- **Schedule**: Every 6 hours (configurable via `DRIFT_CHECK_SCHEDULE`)
- **Action**: Checks all active models for performance drift

## Environment Variables

### Required
- `DATABASE_URL` - PostgreSQL connection string
- `AWS_REGION` - AWS region for S3 and ECS
- `S3_BUCKET_TRAINING` - S3 bucket for training data
- `S3_BUCKET_MODELS` - S3 bucket for model storage
- `ECS_CLUSTER_NAME` - ECS cluster name for training tasks
- `ECS_TASK_DEFINITION` - ECS task definition name
- `ECS_SUBNET_IDS` - Comma-separated subnet IDs
- `ECS_SECURITY_GROUP_IDS` - Comma-separated security group IDs

### Optional
- `TRAINING_SCHEDULE` - Cron schedule for training (default: `0 2 * * 0`)
- `DATA_COLLECTION_SCHEDULE` - Cron schedule for data collection (default: `0 1 * * *`)
- `DRIFT_CHECK_SCHEDULE` - Cron schedule for drift checks (default: `0 */6 * * *`)
- `DRIFT_THRESHOLD` - F1 score degradation threshold (default: `0.05`)
- `MIN_IMPROVEMENT_FOR_DEPLOYMENT` - Minimum F1 improvement to deploy (default: `0.01`)
- `MAX_FALSE_POSITIVE_RATE` - Maximum FPR for deployment (default: `0.02`)

## Python Training Scripts

Training scripts are located in `backend/ml-services/training/`:

- **NLP**: `nlp/train_phishing_model.py`, `nlp/train_ai_detector.py`
- **URL**: `url/train_gnn_model.py`
- **Visual**: `visual/train_cnn_model.py`
- **Validation**: `validation/evaluate_model.py`, `validation/compare_models.py`
- **Data Pipeline**: `data-pipeline/collect_feedback.py`, `data-pipeline/prepare_dataset.py`

## Usage

### Start the Service

```bash
cd backend/core-services/learning-pipeline
npm install
npm run build
npm start
```

### Development Mode

```bash
npm run dev
```

### Manual Training Trigger

The service can be extended to expose API endpoints for manual training triggers.

## Database Tables Used

- `detections` - Detection records
- `detection_feedback` - User feedback on detections
- `threats` - Confirmed threats
- `training_jobs` - Training job tracking
- `ml_models` - Model metadata
- `model_performance` - Model performance metrics

## S3 Structure

### Training Data
```
s3://training-bucket/
  training-data/
    feedback/
      {timestamp}.json
    threats/
      {timestamp}.json
    false-positives/
      {timestamp}.json
  datasets/
    {model-type}/
      {version}/
        dataset.json
        metadata.json
        train.json
        val.json
        test.json
```

### Models
```
s3://models-bucket/
  models/
    {model-type}/
      {version}/
        model.pt (or model files)
        metrics.json
      production/
        model.pt
```

## Monitoring

The service logs all activities:
- Data collection counts
- Training job status
- Model validation results
- Drift detection alerts
- Deployment events

Logs are structured JSON and can be aggregated in CloudWatch or similar services.

## Error Handling

All services include comprehensive error handling:
- Database connection failures are retried
- S3 operations include retry logic
- ECS task failures are logged and tracked
- Training job failures are stored in database

## Future Enhancements

- API endpoints for manual triggers
- WebSocket support for real-time training progress
- MLflow integration for experiment tracking
- Automated hyperparameter tuning
- Multi-region deployment support
