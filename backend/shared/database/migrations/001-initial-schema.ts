import { MigrationInterface, QueryRunner } from 'typeorm';
import * as fs from 'fs';
import * as path from 'path';

export class InitialSchema1234567890 implements MigrationInterface {
  public async up(queryRunner: QueryRunner): Promise<void> {
    // Read and execute schema files in order
    const schemaFiles = [
      'users.sql',
      'threats.sql',
      'domains.sql',
      'ml_models.sql',
      'threat_intel.sql',
      'emails.sql',
      'sandbox.sql',
    ];

    for (const file of schemaFiles) {
      // Try both source and dist paths
      let filePath = path.join(__dirname, '../schemas', file);
      if (!fs.existsSync(filePath)) {
        // If running from dist, try source path
        filePath = path.join(__dirname, '../../database/schemas', file);
      }
      
      if (!fs.existsSync(filePath)) {
        throw new Error(`Schema file not found: ${file}`);
      }

      const sql = fs.readFileSync(filePath, 'utf8');
      
      // Execute the entire SQL file
      // PostgreSQL can handle multiple statements separated by semicolons
      // Remove single-line comments but keep the SQL structure
      const lines = sql.split('\n');
      const cleanedLines = lines
        .map(line => {
          // Remove inline comments (-- comment)
          const commentIndex = line.indexOf('--');
          if (commentIndex >= 0) {
            return line.substring(0, commentIndex).trimEnd();
          }
          return line;
        })
        .filter(line => line.trim().length > 0);
      
      const cleanedSql = cleanedLines.join('\n').trim();
      
      // Execute the entire SQL file at once
      if (cleanedSql) {
        await queryRunner.query(cleanedSql);
      }
    }
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    // Drop all tables in reverse order
    const tables = [
      'sandbox_analyses',
      'email_headers',
      'email_messages',
      'ioc_matches',
      'iocs',
      'threat_intelligence_feeds',
      'model_performance',
      'training_jobs',
      'model_versions',
      'ml_models',
      'domain_relationships',
      'urls',
      'domains',
      'detection_feedback',
      'threat_indicators',
      'detections',
      'threats',
      'api_keys',
      'users',
      'organizations',
    ];

    for (const table of tables) {
      await queryRunner.query(`DROP TABLE IF EXISTS ${table} CASCADE;`);
    }
  }
}
