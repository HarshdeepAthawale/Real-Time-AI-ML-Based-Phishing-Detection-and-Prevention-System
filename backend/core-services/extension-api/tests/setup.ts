// Test setup file - Initialize real database and Redis connections

// IMPORTANT: Set environment variables BEFORE any imports that use them
process.env.NODE_ENV = process.env.NODE_ENV || 'test';
const testDbUrl = process.env.TEST_DATABASE_URL || 'postgresql://postgres:postgres@localhost:5433/phishing_detection_test';
process.env.TEST_DATABASE_URL = testDbUrl;
process.env.DATABASE_URL = testDbUrl; // Set for shared database connection (must be set before importing shared modules)
const testRedisUrl = process.env.TEST_REDIS_URL || 'redis://localhost:6380';
process.env.TEST_REDIS_URL = testRedisUrl;
process.env.REDIS_URL = testRedisUrl; // Set for shared Redis connection
process.env.DETECTION_API_URL = process.env.DETECTION_API_URL || 'http://localhost:3001';
process.env.THREAT_INTEL_URL = process.env.THREAT_INTEL_URL || 'http://localhost:3002';

// Import reflect-metadata for TypeORM decorators
import 'reflect-metadata';

import { connectTestDatabase, disconnectTestDatabase, runMigrations } from './helpers/test-db';
import { connectTestRedis, disconnectTestRedis, flushTestRedis } from './helpers/test-redis';
import { seedTestDatabase } from './fixtures/database';
// Import shared database utilities
// Note: Using require to avoid TypeScript path resolution issues in Jest
const sharedDbModule = require('../../../shared/database');
const connectPostgreSQL = sharedDbModule.connectPostgreSQL;
const disconnectPostgreSQL = sharedDbModule.disconnectPostgreSQL;

// Global setup - runs once before all tests
beforeAll(async () => {
  try {
    // Wait a bit for Docker containers to be fully ready
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Connect to test database using shared utilities (for middleware)
    // Retry connection up to 3 times
    let retries = 3;
    while (retries > 0) {
      try {
        await connectPostgreSQL();
        break;
      } catch (error: any) {
        retries--;
        if (retries === 0) throw error;
        console.warn(`Database connection failed, retrying... (${retries} attempts left)`);
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    
    // Also connect using test helpers (for test utilities)
    const dataSource = await connectTestDatabase();
    
    // Run migrations
    await runMigrations();
    
    // Seed database
    await seedTestDatabase(dataSource);
    
    // Connect to test Redis
    // Retry connection up to 3 times
    retries = 3;
    while (retries > 0) {
      try {
        await connectTestRedis();
        break;
      } catch (error: any) {
        retries--;
        if (retries === 0) throw error;
        console.warn(`Redis connection failed, retrying... (${retries} attempts left)`);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    console.log('Test infrastructure initialized');
  } catch (error) {
    console.error('Failed to initialize test infrastructure:', error);
    // Don't throw - allow tests to run even if infrastructure setup fails
    // Individual tests will handle their own connection requirements
  }
}, 60000); // 60 second timeout for setup

// Global teardown - runs once after all tests
afterAll(async () => {
  try {
    // Disconnect from test database (both connections)
    await disconnectTestDatabase();
    await disconnectPostgreSQL();
    
    // Disconnect from test Redis
    await disconnectTestRedis();
    
    console.log('Test infrastructure cleaned up');
  } catch (error) {
    console.error('Error during test cleanup:', error);
  }
}, 10000);

// Increase timeout for integration tests
jest.setTimeout(30000);
