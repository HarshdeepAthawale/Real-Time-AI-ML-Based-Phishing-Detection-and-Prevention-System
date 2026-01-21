import { DataSource } from 'typeorm';
import { truncateTables, runMigrations } from '../helpers/test-db';

/**
 * Seed test database with initial data
 */
export const seedTestDatabase = async (dataSource: DataSource): Promise<void> => {
  // Run migrations first to ensure schema exists
  await runMigrations();
  
  // Additional seed data can be added here if needed
  // For now, migrations create the schema and fixtures create the data
};

/**
 * Clean up test database (truncate all tables)
 */
export const cleanupTestDatabase = async (dataSource: DataSource): Promise<void> => {
  await truncateTables();
};

/**
 * Reset test database (drop and recreate)
 */
export const resetTestDatabase = async (dataSource: DataSource): Promise<void> => {
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
