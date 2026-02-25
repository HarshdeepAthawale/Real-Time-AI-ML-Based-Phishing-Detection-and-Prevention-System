import { execFile } from 'child_process';
import path from 'path';
import { Config } from '../../../../shared/config';
import { logger } from '../utils/logger';

interface TrainingStatus {
  modelType: string;
  status: 'idle' | 'running' | 'completed' | 'failed';
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

/**
 * Local training service that executes Python training scripts
 * directly on the host machine, without requiring AWS ECS.
 *
 * Uses the same training scripts that live alongside each ML service:
 *   nlp:    ml-services/nlp-service/scripts/train_phishing_model.py
 *   url:    ml-services/url-service/scripts/create_gnn_model.py
 *   visual: ml-services/visual-service/scripts/train_cnn_classifier.py
 */
export class LocalTrainingService {
  private config: Config;
  private currentStatus: TrainingStatus = {
    modelType: '',
    status: 'idle',
  };

  constructor(config: Config) {
    this.config = config;
  }

  getStatus(): TrainingStatus {
    return { ...this.currentStatus };
  }

  async triggerTraining(modelType: string): Promise<{ status: string; message: string }> {
    if (this.currentStatus.status === 'running') {
      return {
        status: 'already_running',
        message: `Training already in progress for model: ${this.currentStatus.modelType}`,
      };
    }

    const scriptInfo = this.getTrainingScript(modelType);
    if (!scriptInfo) {
      return { status: 'error', message: `Unknown model type: ${modelType}` };
    }

    this.currentStatus = {
      modelType,
      status: 'running',
      startedAt: new Date().toISOString(),
    };

    // Run training asynchronously (don't block the API response)
    this.runTraining(scriptInfo.script, scriptInfo.args, modelType).catch((err) => {
      logger.error(`Local training failed for ${modelType}: ${err.message}`);
    });

    return {
      status: 'started',
      message: `Local training started for ${modelType}. Check /api/learning/status for progress.`,
    };
  }

  private getTrainingScript(modelType: string): { script: string; args: string[] } | null {
    // Resolve paths relative to the project root (backend/ml-services/...)
    const mlServicesRoot = path.resolve(__dirname, '../../../../ml-services');

    const scripts: Record<string, { script: string; args: string[] }> = {
      nlp: {
        script: path.join(mlServicesRoot, 'nlp-service', 'scripts', 'train_phishing_model.py'),
        args: [
          '--output-dir', path.join(mlServicesRoot, 'nlp-service', 'models', 'phishing-detector'),
          '--epochs', '3',
          '--batch-size', '16',
        ],
      },
      url: {
        script: path.join(mlServicesRoot, 'url-service', 'scripts', 'create_gnn_model.py'),
        args: [],
      },
      visual: {
        script: path.join(mlServicesRoot, 'visual-service', 'scripts', 'train_cnn_classifier.py'),
        args: [
          '--output-dir', path.join(mlServicesRoot, 'visual-service', 'models', 'brand-classifier'),
          '--epochs', '5',
        ],
      },
    };

    return scripts[modelType] || null;
  }

  private runTraining(scriptPath: string, args: string[], modelType: string): Promise<void> {
    return new Promise((resolve, reject) => {
      logger.info(`Starting local training: python ${scriptPath} ${args.join(' ')}`);

      const proc = execFile('python', [scriptPath, ...args], {
        timeout: 30 * 60 * 1000, // 30 minute timeout
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
        },
      }, (error, stdout, stderr) => {
        if (error) {
          this.currentStatus = {
            modelType,
            status: 'failed',
            startedAt: this.currentStatus.startedAt,
            completedAt: new Date().toISOString(),
            error: error.message,
          };
          logger.error(`Local training failed for ${modelType}: ${error.message}`);
          if (stderr) logger.error(`stderr: ${stderr}`);
          reject(error);
          return;
        }

        this.currentStatus = {
          modelType,
          status: 'completed',
          startedAt: this.currentStatus.startedAt,
          completedAt: new Date().toISOString(),
        };
        logger.info(`Local training completed for ${modelType}`);
        if (stdout) logger.info(`stdout: ${stdout.slice(-500)}`);
        resolve();
      });
    });
  }
}
