import { DataSource } from 'typeorm';
import { config } from '../config';
import { logger } from '../utils/logger';
import { IOC } from '../../../../shared/database/models/IOC';
import { ThreatIntelligenceFeed } from '../../../../shared/database/models/ThreatIntelligenceFeed';
import { IOCMatch } from '../../../../shared/database/models/IOCMatch';

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
    url: config.database.url,
    entities: [IOC, ThreatIntelligenceFeed, IOCMatch],
    synchronize: false, // We use migrations
    logging: config.nodeEnv === 'development',
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
