#!/usr/bin/env node
import 'reflect-metadata';
import { connectAllDatabases, disconnectAllDatabases, getPostgreSQL } from './connection';
import { getMongoDB } from './mongodb/connection';
import { getRedis } from './redis/connection';

async function verifyDatabaseSetup() {
  try {
    console.log('Connecting to all databases...');
    await connectAllDatabases();

    // Verify PostgreSQL
    console.log('\n✓ Verifying PostgreSQL...');
    const pg = getPostgreSQL();
    const result = await pg.query('SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = \'public\'');
    console.log(`  Found ${result[0].count} tables in PostgreSQL`);

    // Verify MongoDB
    console.log('\n✓ Verifying MongoDB...');
    const mongo = getMongoDB();
    const collections = await mongo.listCollections().toArray();
    console.log(`  Found ${collections.length} collections in MongoDB`);

    // Verify Redis
    console.log('\n✓ Verifying Redis...');
    const redis = getRedis();
    await redis.ping();
    console.log('  Redis connection successful');

    console.log('\n✅ All databases verified successfully!');
    
    await disconnectAllDatabases();
    process.exit(0);
  } catch (error) {
    console.error('\n❌ Database verification failed:', error);
    process.exit(1);
  }
}

verifyDatabaseSetup();
