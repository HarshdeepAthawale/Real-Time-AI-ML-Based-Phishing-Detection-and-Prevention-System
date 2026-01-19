#!/usr/bin/env node
import 'reflect-metadata';
import { AppDataSource } from './data-source';

async function runMigrations() {
  try {
    console.log('Initializing database connection...');
    await AppDataSource.initialize();

    console.log('Running migrations...');
    await AppDataSource.runMigrations();

    console.log('Migrations completed successfully');
    await AppDataSource.destroy();
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  }
}

runMigrations();
