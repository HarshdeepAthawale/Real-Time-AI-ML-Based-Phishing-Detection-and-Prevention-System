import { DataSource } from 'typeorm';
import * as entities from '../../../../shared/database/models';
import * as path from 'path';

let testDataSource: DataSource | null = null;

/**
 * Get test database connection URL from environment
 */
export const getTestDatabaseUrl = (): string => {
  return process.env.TEST_DATABASE_URL || 'postgresql://postgres:postgres@localhost:5433/phishing_detection_test';
};

/**
 * Connect to test database
 */
export const connectTestDatabase = async (): Promise<DataSource> => {
  if (testDataSource?.isInitialized) {
    return testDataSource;
  }

  const databaseUrl = getTestDatabaseUrl();
  
  // Get migrations path - try both relative and absolute
  const migrationsPath = path.join(__dirname, '../../../../shared/database/migrations/**/*.ts');
  
  testDataSource = new DataSource({
    type: 'postgres',
    url: databaseUrl,
    entities: Object.values(entities),
    migrations: [migrationsPath],
    synchronize: true, // Use synchronize for tests to auto-create schema
    logging: false, // Disable logging in tests
    extra: {
      max: 10,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 5000,
    },
  });

  try {
    await testDataSource.initialize();
    return testDataSource;
  } catch (error) {
    console.error('Failed to connect to test database:', error);
    throw error;
  }
};

/**
 * Get test database connection
 */
export const getTestDatabase = (): DataSource => {
  if (!testDataSource?.isInitialized) {
    throw new Error('Test database not connected. Call connectTestDatabase() first.');
  }
  return testDataSource;
};

/**
 * Disconnect from test database
 */
export const disconnectTestDatabase = async (): Promise<void> => {
  if (testDataSource?.isInitialized) {
    await testDataSource.destroy();
    testDataSource = null;
  }
};

/**
 * Run database migrations
 * Note: With synchronize: true, migrations aren't needed, but we keep this for compatibility
 */
export const runMigrations = async (): Promise<void> => {
  const dataSource = await connectTestDatabase();
  // With synchronize: true, schema is auto-created from entities
  // No need to run migrations manually
  // But we can still try if needed
  try {
    const migrations = await dataSource.runMigrations();
    if (migrations.length > 0) {
      console.log(`Ran ${migrations.length} migrations`);
    }
  } catch (error: any) {
    // Ignore migration errors if synchronize is enabled
    // Schema will be created from entities automatically
    console.log('Migrations skipped (using synchronize)');
  }
};

/**
 * Revert all migrations
 */
export const revertMigrations = async (): Promise<void> => {
  const dataSource = await connectTestDatabase();
  await dataSource.undoLastMigration();
};

/**
 * Truncate all tables (cleanup between tests)
 * Handles foreign key constraints by truncating in correct order
 */
export const truncateTables = async (): Promise<void> => {
  const dataSource = await connectTestDatabase();
  const queryRunner = dataSource.createQueryRunner();
  
  try {
    // Get all table names
    const tables = await queryRunner.query(`
      SELECT tablename 
      FROM pg_tables 
      WHERE schemaname = 'public'
      ORDER BY tablename
    `);
    
    // Disable foreign key checks temporarily using session_replication_role
    // This allows truncating tables in any order without FK violations
    await queryRunner.query('SET session_replication_role = replica;');
    
    // Truncate all tables with CASCADE to handle dependencies
    for (const table of tables) {
      try {
        await queryRunner.query(`TRUNCATE TABLE "${table.tablename}" CASCADE;`);
      } catch (error: any) {
        // Ignore errors for tables that don't exist or are already empty
        // This can happen if tables are created/dropped dynamically
        if (!error.message.includes('does not exist') && !error.message.includes('relation')) {
          console.warn(`Warning: Could not truncate table ${table.tablename}:`, error.message);
        }
      }
    }
    
    // Re-enable foreign key checks
    await queryRunner.query('SET session_replication_role = DEFAULT;');
  } catch (error) {
    console.error('Error truncating tables:', error);
    throw error;
  } finally {
    await queryRunner.release();
  }
};

/**
 * Reset database (drop and recreate schema)
 */
export const resetDatabase = async (): Promise<void> => {
  const dataSource = await connectTestDatabase();
  const queryRunner = dataSource.createQueryRunner();
  
  try {
    // Drop all tables
    await queryRunner.query(`
      DO $$ DECLARE
        r RECORD;
      BEGIN
        FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
          EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
        END LOOP;
      END $$;
    `);
    
    // Run migrations to recreate schema
    await runMigrations();
  } finally {
    await queryRunner.release();
  }
};
