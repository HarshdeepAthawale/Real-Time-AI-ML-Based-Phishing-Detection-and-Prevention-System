#!/usr/bin/env ts-node
/**
 * Initial Setup Script
 * 
 * This script helps create the first organization, admin user, and API key.
 * Run this ONCE after initial database setup.
 * 
 * Usage:
 *   ts-node create-initial-setup.ts \
 *     --org-name "Your Organization" \
 *     --org-domain "yourdomain.com" \
 *     --admin-email "admin@yourdomain.com" \
 *     --admin-password "your-secure-password" \
 *     --admin-name "Admin User"
 */

import 'reflect-metadata';
import { DataSource } from 'typeorm';
import * as bcrypt from 'bcryptjs';
import * as crypto from 'crypto';
import * as yargs from 'yargs';

// Parse command line arguments
const argv = yargs
  .option('org-name', {
    type: 'string',
    description: 'Organization name',
    demandOption: true
  })
  .option('org-domain', {
    type: 'string',
    description: 'Organization domain',
    demandOption: true
  })
  .option('admin-email', {
    type: 'string',
    description: 'Admin email',
    demandOption: true
  })
  .option('admin-password', {
    type: 'string',
    description: 'Admin password',
    demandOption: true
  })
  .option('admin-name', {
    type: 'string',
    description: 'Admin full name',
    default: 'Admin User'
  })
  .option('tier', {
    type: 'string',
    description: 'Subscription tier',
    choices: ['free', 'professional', 'enterprise'],
    default: 'enterprise'
  })
  .help()
  .parseSync();

async function createInitialSetup() {
  // Database connection
  const dataSource = new DataSource({
    type: 'postgres',
    host: process.env.POSTGRES_HOST || 'localhost',
    port: parseInt(process.env.POSTGRES_PORT || '5432'),
    username: process.env.POSTGRES_USER || 'postgres',
    password: process.env.POSTGRES_PASSWORD || '',
    database: process.env.POSTGRES_DB || 'phishing_detection',
    synchronize: false,
    logging: false
  });

  try {
    console.log('🔌 Connecting to database...');
    await dataSource.initialize();
    console.log('✓ Connected');

    // Check if organization already exists
    const existingOrg = await dataSource.query(
      'SELECT id FROM organizations LIMIT 1'
    );

    if (existingOrg.length > 0) {
      console.error('❌ Error: Organizations already exist. This script should only be run once.');
      console.log('   To create additional organizations, use the API endpoints.');
      process.exit(1);
    }

    // Create organization
    console.log('\n📊 Creating organization...');
    const orgResult = await dataSource.query(
      `INSERT INTO organizations (name, domain, subscription_tier, is_active)
       VALUES ($1, $2, $3, true)
       RETURNING id, name`,
      [argv['org-name'], argv['org-domain'], argv.tier]
    );
    const organizationId = orgResult[0].id;
    console.log(`✓ Organization created: ${orgResult[0].name} (${organizationId})`);

    // Hash password
    console.log('\n🔐 Creating admin user...');
    const passwordHash = await bcrypt.hash(argv['admin-password'], 10);
    const [firstName, ...lastNameParts] = argv['admin-name'].split(' ');
    const lastName = lastNameParts.join(' ') || '';

    // Create admin user
    const userResult = await dataSource.query(
      `INSERT INTO users (organization_id, email, password_hash, first_name, last_name, role, is_active)
       VALUES ($1, $2, $3, $4, $5, 'admin', true)
       RETURNING id, email`,
      [organizationId, argv['admin-email'], passwordHash, firstName, lastName]
    );
    const userId = userResult[0].id;
    console.log(`✓ Admin user created: ${userResult[0].email} (${userId})`);

    // Generate API key
    console.log('\n🔑 Creating API key...');
    const apiKey = crypto.randomBytes(32).toString('hex');
    const keyHash = await bcrypt.hash(apiKey, 10);

    const apiKeyResult = await dataSource.query(
      `INSERT INTO api_keys (organization_id, user_id, key_hash, name, permissions)
       VALUES ($1, $2, $3, $4, $5::jsonb)
       RETURNING id, name`,
      [
        organizationId,
        userId,
        keyHash,
        'Primary API Key',
        JSON.stringify(['read', 'write', 'admin'])
      ]
    );
    console.log(`✓ API key created: ${apiKeyResult[0].name} (${apiKeyResult[0].id})`);

    // Activate threat intelligence feeds if API keys are configured
    console.log('\n🛡️  Checking threat intelligence configuration...');
    const feedsToActivate = [];

    if (process.env.MISP_URL && process.env.MISP_API_KEY) {
      feedsToActivate.push('MISP Feed');
    }
    if (process.env.OTX_API_KEY) {
      feedsToActivate.push('AlienVault OTX');
    }
    if (process.env.PHISHTANK_API_KEY) {
      feedsToActivate.push('PhishTank');
    }
    if (process.env.VIRUSTOTAL_API_KEY) {
      feedsToActivate.push('VirusTotal');
    }
    // URLhaus doesn't require API key
    feedsToActivate.push('URLhaus');

    if (feedsToActivate.length > 0) {
      await dataSource.query(
        `UPDATE threat_intelligence_feeds 
         SET is_active = true 
         WHERE name = ANY($1::text[])`,
        [feedsToActivate]
      );
      console.log(`✓ Activated ${feedsToActivate.length} threat intelligence feeds:`);
      feedsToActivate.forEach(feed => console.log(`  - ${feed}`));
    } else {
      console.log('⚠️  No threat intelligence API keys configured');
      console.log('   Add API keys to .env and restart services to enable feeds');
    }

    // Display summary
    console.log('\n' + '='.repeat(70));
    console.log('🎉 INITIAL SETUP COMPLETE');
    console.log('='.repeat(70));
    console.log('\n📋 Your Credentials:');
    console.log(`   Organization: ${argv['org-name']}`);
    console.log(`   Admin Email:  ${argv['admin-email']}`);
    console.log(`   Admin Pass:   [provided]`);
    console.log(`\n🔑 Your API Key (SAVE THIS - shown only once):`);
    console.log(`   ${apiKey}`);
    console.log('\n📚 Next Steps:');
    console.log('   1. Save your API key securely');
    console.log('   2. Add threat intel API keys to .env');
    console.log('   3. Start all services: docker-compose up -d');
    console.log('   4. Test detection: curl -X POST http://localhost:3001/api/v1/detect/url \\');
    console.log(`                           -H "X-API-Key: ${apiKey}" \\`);
    console.log('                           -H "Content-Type: application/json" \\');
    console.log('                           -d \'{"url": "http://example.com"}\'');
    console.log('\n' + '='.repeat(70));

    await dataSource.destroy();
    process.exit(0);

  } catch (error) {
    console.error('\n❌ Setup failed:', error);
    await dataSource.destroy();
    process.exit(1);
  }
}

// Run setup
createInitialSetup();
