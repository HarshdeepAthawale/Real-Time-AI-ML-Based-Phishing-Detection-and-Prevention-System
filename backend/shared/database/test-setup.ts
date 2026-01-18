// Test database setup utilities

import { Pool } from 'pg';

let testPool: Pool | null = null;

export const getTestDatabase = async (): Promise<Pool> => {
  if (!testPool) {
    testPool = new Pool({
      connectionString: process.env.TEST_DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/phishing_detection_test',
    });
  }
  return testPool;
};

export const cleanupTestDatabase = async (): Promise<void> => {
  if (testPool) {
    await testPool.end();
    testPool = null;
  }
};

export const resetTestDatabase = async (): Promise<void> => {
  const pool = await getTestDatabase();
  
  // Drop all tables
  await pool.query(`
    DO $$ DECLARE
      r RECORD;
    BEGIN
      FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
      END LOOP;
    END $$;
  `);
};
