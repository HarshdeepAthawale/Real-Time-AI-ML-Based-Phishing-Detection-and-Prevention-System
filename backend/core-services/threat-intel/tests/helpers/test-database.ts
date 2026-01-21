import { DataSource, QueryRunner } from 'typeorm';
// Import all entities dynamically from models index (non-hardcoded solution)
// This ensures all entities in relationship chains are included automatically
import * as allEntities from '../../../../shared/database/models';

// Use all entities from the models index - this is the same approach used by
// the shared database connection, ensuring consistency and avoiding hardcoded lists
const entities = Object.values(allEntities);

// Mock logger for tests to avoid console output
const logger = {
  info: jest.fn(),
  error: jest.fn(),
  warn: jest.fn(),
  debug: jest.fn(),
};

let testDataSource: DataSource | null = null;
let databaseAvailable: boolean | null = null;
let activeQueryRunner: QueryRunner | null = null;

// Reset cache function for testing
export function resetDatabaseAvailabilityCache(): void {
  databaseAvailable = null;
}

/**
 * Check if database is available
 */
export async function checkDatabaseAvailability(): Promise<boolean> {
  if (databaseAvailable !== null) {
    return databaseAvailable;
  }

  // Get password from environment or use default postgres
  const postgresPassword = process.env.POSTGRES_PASSWORD || 
    process.env.DATABASE_URL?.match(/:\/\/([^:]+):([^@]+)@/)?.[2] || 
    'postgres';
  
  // Construct test database URL
  let testDbUrl = process.env.TEST_DATABASE_URL;
  
  if (!testDbUrl && process.env.DATABASE_URL) {
    // Replace database name in existing DATABASE_URL
    testDbUrl = process.env.DATABASE_URL.replace(/phishing_detection/, 'phishing_detection_test');
    // Also ensure user is postgres if it's not already set correctly
    testDbUrl = testDbUrl.replace(/:\/\/([^:]+):/, `://postgres:`);
  }
  
  if (!testDbUrl) {
    // Default fallback
    testDbUrl = `postgresql://postgres:${postgresPassword}@localhost:5432/phishing_detection_test`;
  }

  const tempDataSource = new DataSource({
    type: 'postgres',
    url: testDbUrl,
    entities: [], // Empty entities for availability check - just testing connection
    synchronize: false,
    logging: false,
    extra: {
      max: 1,
      idleTimeoutMillis: 5000,
      connectionTimeoutMillis: 5000, // Increased timeout
    },
  });

  try {
    await tempDataSource.initialize();
    await tempDataSource.destroy();
    databaseAvailable = true;
    return true;
  } catch (error: any) {
    databaseAvailable = false;
    // Log error for debugging (only in test environment)
    if (process.env.NODE_ENV === 'test' || process.env.DEBUG_TESTS) {
      console.error('Database availability check failed:', error.message, 'URL:', testDbUrl.replace(/:[^:@]+@/, ':****@'));
    }
    return false;
  }
}

/**
 * Setup test database connection
 */
export async function setupTestDatabase(): Promise<DataSource> {
  if (testDataSource?.isInitialized) {
    return testDataSource;
  }

  // Check availability first
  const isAvailable = await checkDatabaseAvailability();
  if (!isAvailable) {
    const testDbUrl = process.env.TEST_DATABASE_URL || 
      process.env.DATABASE_URL?.replace(/phishing_detection/, 'phishing_detection_test') ||
      'postgresql://postgres:postgres@localhost:5432/phishing_detection_test';
    
    throw new Error(
      `Test database is not available at ${testDbUrl.replace(/:[^:@]+@/, ':****@')}. ` +
      `\n\nTo run integration tests:` +
      `\n1. Make sure PostgreSQL is running` +
      `\n2. Create test database: createdb phishing_detection_test` +
      `\n3. Run migrations: cd backend/shared/database && npm run migrate` +
      `\n4. Set TEST_DATABASE_URL if using different connection string` +
      `\n\nIntegration tests will be skipped if database is unavailable.`
    );
  }

  // Get password from environment or use default postgres
  const postgresPassword = process.env.POSTGRES_PASSWORD || 
    process.env.DATABASE_URL?.match(/:\/\/([^:]+):([^@]+)@/)?.[2] || 
    'postgres';
  
  // Construct test database URL
  let testDbUrl = process.env.TEST_DATABASE_URL;
  
  if (!testDbUrl && process.env.DATABASE_URL) {
    // Replace database name in existing DATABASE_URL
    testDbUrl = process.env.DATABASE_URL.replace(/phishing_detection/, 'phishing_detection_test');
    // Also ensure user is postgres if it's not already set correctly
    testDbUrl = testDbUrl.replace(/:\/\/([^:]+):/, `://postgres:`);
  }
  
  if (!testDbUrl) {
    // Default fallback
    testDbUrl = `postgresql://postgres:${postgresPassword}@localhost:5432/phishing_detection_test`;
  }

  testDataSource = new DataSource({
    type: 'postgres',
    url: testDbUrl,
    entities: entities, // Use threat intel entities
    synchronize: false, // We use migrations
    logging: false, // Disable logging in tests
    extra: {
      max: 10, // Fewer connections for tests
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    },
  });

  try {
    await testDataSource.initialize();
    logger.info('Test database connected');
    return testDataSource;
  } catch (error: any) {
    const errorMessage = error?.message || String(error);
    const errorCode = error?.code;
    
    // Log the actual error for debugging
    console.error('Database connection error:', {
      message: errorMessage,
      code: errorCode,
      url: testDbUrl.replace(/:[^:@]+@/, ':****@'),
    });
    
    // Provide helpful error message for connection issues
    if (errorCode === 'ECONNREFUSED' || errorMessage.includes('ECONNREFUSED')) {
      throw new Error(
        `Cannot connect to test database at ${testDbUrl.replace(/:[^:@]+@/, ':****@')}. ` +
        `Make sure PostgreSQL is running and accessible. ` +
        `Set TEST_DATABASE_URL environment variable if using a different database. ` +
        `\n\nIntegration tests will be skipped if database is unavailable.`
      );
    }
    
    // For password/auth errors, re-throw with more context
    if (errorMessage.includes('password') || errorMessage.includes('authentication') || 
        errorCode === '28P01' || errorCode === '28000') {
      throw new Error(
        `Authentication failed for test database. ` +
        `Please check POSTGRES_PASSWORD environment variable or TEST_DATABASE_URL. ` +
        `Original error: ${errorMessage}`
      );
    }
    
    // Re-throw original error for other issues
    throw error;
  }
}

/**
 * Get test database connection
 */
export function getTestDataSource(): DataSource {
  if (!testDataSource?.isInitialized) {
    throw new Error('Test database not connected. Call setupTestDatabase() first.');
  }
  return testDataSource;
}

/**
 * Clean test database - truncate all tables
 */
export async function cleanupTestDatabase(): Promise<void> {
  if (!testDataSource?.isInitialized) {
    return;
  }

  try {
    // Disable foreign key checks temporarily to allow truncation
    await testDataSource.query('SET session_replication_role = replica;');
    
    try {
      // Truncate tables in reverse dependency order
      await testDataSource.query('TRUNCATE TABLE ioc_matches CASCADE;');
      await testDataSource.query('TRUNCATE TABLE iocs CASCADE;');
      await testDataSource.query('TRUNCATE TABLE threat_intelligence_feeds CASCADE;');
    } catch (error: any) {
      // Log the error but don't fail - tables might not exist
      const errorMessage = error?.message || String(error);
      if (!errorMessage.includes('does not exist') && !errorMessage.includes('relation')) {
        logger.debug('Error cleaning test database', error);
        // Re-throw if it's not a "table doesn't exist" error
        throw error;
      }
    } finally {
      // Re-enable foreign key checks
      await testDataSource.query('SET session_replication_role = DEFAULT;');
    }
  } catch (error) {
    // Log but don't fail - allows tests to continue
    logger.debug('Database cleanup failed', error);
  }
}

/**
 * Begin a test transaction
 * Each test should start a transaction in beforeEach and rollback in afterEach
 */
export async function beginTestTransaction(): Promise<QueryRunner> {
  if (!testDataSource?.isInitialized) {
    throw new Error('Test database not initialized. Call setupTestDatabase() first.');
  }
  
  // If there's already an active transaction, roll it back first
  if (activeQueryRunner) {
    await rollbackTestTransaction();
  }
  
  const queryRunner = testDataSource.createQueryRunner();
  await queryRunner.connect();
  await queryRunner.startTransaction();
  activeQueryRunner = queryRunner;
  return queryRunner;
}

/**
 * Rollback the active test transaction
 * This should be called in afterEach to clean up test data
 */
export async function rollbackTestTransaction(): Promise<void> {
  if (activeQueryRunner) {
    try {
      await activeQueryRunner.rollbackTransaction();
    } catch (error) {
      // Log but don't throw - transaction might already be rolled back
      logger.debug('Error rolling back transaction', error);
    } finally {
      try {
        await activeQueryRunner.release();
      } catch (error) {
        logger.debug('Error releasing query runner', error);
      }
      activeQueryRunner = null;
    }
  }
}

/**
 * Get the active query runner for the current test
 * Returns null if no transaction is active
 */
export function getTestQueryRunner(): QueryRunner | null {
  return activeQueryRunner;
}

/**
 * Disconnect from test database
 */
export async function disconnectTestDatabase(): Promise<void> {
  // Rollback any active transaction before disconnecting
  if (activeQueryRunner) {
    await rollbackTestTransaction();
  }
  
  if (testDataSource?.isInitialized) {
    await testDataSource.destroy();
    testDataSource = null;
    logger.info('Test database disconnected');
  }
}
