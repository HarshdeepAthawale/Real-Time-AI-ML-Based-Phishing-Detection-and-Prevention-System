import 'reflect-metadata';
import { DataSource } from 'typeorm';
import { config } from '../config';
import { logger } from '../utils/logger';
import { SandboxAnalysis } from '../../../../shared/database/models/SandboxAnalysis';
import { Threat } from '../../../../shared/database/models/Threat';
import { Organization } from '../../../../shared/database/models/Organization';

let dataSource: DataSource | null = null;

/**
 * Initialize database connection using TypeORM
 */
export async function connectDatabase(): Promise<DataSource> {
  if (dataSource?.isInitialized) {
    return dataSource;
  }

  dataSource = new DataSource({
    type: 'postgres',
    url: process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/phishing_detection',
    entities: [SandboxAnalysis, Threat, Organization],
    synchronize: false, // We use migrations
    logging: process.env.NODE_ENV === 'development',
    extra: {
      max: 20, // Maximum number of connections in the pool
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    },
  });

  try {
    await dataSource.initialize();
    logger.info('PostgreSQL connected via TypeORM');
    return dataSource;
  } catch (error) {
    logger.error('Failed to connect to PostgreSQL', error);
    throw error;
  }
}

/**
 * Get the database connection
 */
export function getDatabase(): DataSource {
  if (!dataSource?.isInitialized) {
    throw new Error('Database not connected. Call connectDatabase() first.');
  }
  return dataSource;
}

/**
 * Disconnect from database
 */
export async function disconnectDatabase(): Promise<void> {
  if (dataSource?.isInitialized) {
    await dataSource.destroy();
    dataSource = null;
    logger.info('PostgreSQL disconnected');
  }
}
