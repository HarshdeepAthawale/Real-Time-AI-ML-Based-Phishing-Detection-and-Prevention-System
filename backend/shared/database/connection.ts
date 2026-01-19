import { DataSource } from 'typeorm';
import { config } from '../config';
import * as entities from './models';
import { connectMongoDB, disconnectMongoDB } from './mongodb/connection';
import { connectRedis, disconnectRedis } from './redis/connection';

let postgresDataSource: DataSource | null = null;

export const connectPostgreSQL = async (): Promise<DataSource> => {
  if (postgresDataSource?.isInitialized) {
    return postgresDataSource;
  }

  postgresDataSource = new DataSource({
    type: 'postgres',
    url: config.database.url,
    entities: Object.values(entities),
    synchronize: false, // We use migrations
    logging: config.app.nodeEnv === 'development',
    extra: {
      max: 20, // Maximum number of connections in the pool
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    },
  });

  try {
    await postgresDataSource.initialize();
    console.log('PostgreSQL connected via TypeORM');
    return postgresDataSource;
  } catch (error) {
    console.error('Failed to connect to PostgreSQL:', error);
    throw error;
  }
};

export const getPostgreSQL = (): DataSource => {
  if (!postgresDataSource?.isInitialized) {
    throw new Error('PostgreSQL not connected. Call connectPostgreSQL() first.');
  }
  return postgresDataSource;
};

export const disconnectPostgreSQL = async (): Promise<void> => {
  if (postgresDataSource?.isInitialized) {
    await postgresDataSource.destroy();
    postgresDataSource = null;
  }
};

// Connect all databases
export const connectAllDatabases = async (): Promise<void> => {
  await Promise.all([
    connectPostgreSQL(),
    connectMongoDB(),
    connectRedis(),
  ]);
  console.log('All databases connected');
};

// Disconnect all databases
export const disconnectAllDatabases = async (): Promise<void> => {
  await Promise.all([
    disconnectPostgreSQL(),
    disconnectMongoDB(),
    disconnectRedis(),
  ]);
  console.log('All databases disconnected');
};
