# Database Backup Strategy

## Overview

This document outlines the backup and recovery strategy for all databases in the Real-Time AI/ML-Based Phishing Detection and Prevention System.

## Backup Requirements

### PostgreSQL

**Backup Frequency:**
- **Full Backup**: Daily at 2:00 AM UTC
- **Incremental Backup**: Every 6 hours
- **Transaction Log Backup**: Continuous (WAL archiving)

**Retention Policy:**
- Daily backups: 30 days
- Weekly backups: 12 weeks
- Monthly backups: 12 months

**Backup Method:**
```bash
# Full backup using pg_dump
pg_dump -h localhost -U postgres -d phishing_detection \
  --format=custom \
  --file=/backups/postgres/full_$(date +%Y%m%d_%H%M%S).dump

# With compression
pg_dump -h localhost -U postgres -d phishing_detection \
  --format=custom \
  --compress=9 \
  --file=/backups/postgres/full_$(date +%Y%m%d_%H%M%S).dump
```

**WAL Archiving:**
```postgresql
-- In postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /backups/postgres/wal/%f'
```

**Automated Backup Script:**
```bash
#!/bin/bash
# scripts/backup-postgres.sh

BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="phishing_detection"

# Create backup directory
mkdir -p $BACKUP_DIR

# Full backup
pg_dump -h localhost -U postgres -d $DB_NAME \
  --format=custom \
  --compress=9 \
  --file=$BACKUP_DIR/full_${DATE}.dump

# Upload to S3 (if configured)
if [ -n "$S3_BACKUP_BUCKET" ]; then
  aws s3 cp $BACKUP_DIR/full_${DATE}.dump \
    s3://$S3_BACKUP_BUCKET/postgres/full_${DATE}.dump
fi

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "full_*.dump" -mtime +30 -delete
```

### MongoDB

**Backup Frequency:**
- **Full Backup**: Daily at 3:00 AM UTC
- **Incremental Backup**: Every 6 hours (using oplog)

**Retention Policy:**
- Daily backups: 30 days
- Weekly backups: 12 weeks

**Backup Method:**
```bash
# Full backup using mongodump
mongodump --uri="mongodb://localhost:27017/phishing_detection" \
  --out=/backups/mongodb/full_$(date +%Y%m%d_%H%M%S)

# With compression
mongodump --uri="mongodb://localhost:27017/phishing_detection" \
  --archive=/backups/mongodb/full_$(date +%Y%m%d_%H%M%S).archive \
  --gzip
```

**Oplog Backup (for point-in-time recovery):**
```bash
# Backup oplog for incremental backups
mongodump --uri="mongodb://localhost:27017/local" \
  --collection=oplog.rs \
  --archive=/backups/mongodb/oplog_$(date +%Y%m%d_%H%M%S).archive \
  --gzip
```

**Automated Backup Script:**
```bash
#!/bin/bash
# scripts/backup-mongodb.sh

BACKUP_DIR="/backups/mongodb"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="phishing_detection"

# Create backup directory
mkdir -p $BACKUP_DIR

# Full backup with compression
mongodump --uri="mongodb://localhost:27017/$DB_NAME" \
  --archive=$BACKUP_DIR/full_${DATE}.archive \
  --gzip

# Oplog backup
mongodump --uri="mongodb://localhost:27017/local" \
  --collection=oplog.rs \
  --archive=$BACKUP_DIR/oplog_${DATE}.archive \
  --gzip

# Upload to S3 (if configured)
if [ -n "$S3_BACKUP_BUCKET" ]; then
  aws s3 cp $BACKUP_DIR/full_${DATE}.archive \
    s3://$S3_BACKUP_BUCKET/mongodb/full_${DATE}.archive
  aws s3 cp $BACKUP_DIR/oplog_${DATE}.archive \
    s3://$S3_BACKUP_BUCKET/mongodb/oplog_${DATE}.archive
fi

# Cleanup old backups
find $BACKUP_DIR -name "full_*.archive" -mtime +30 -delete
find $BACKUP_DIR -name "oplog_*.archive" -mtime +7 -delete
```

### Redis

**Backup Frequency:**
- **RDB Snapshot**: Every 6 hours
- **AOF (Append-Only File)**: Continuous (if enabled)

**Retention Policy:**
- RDB snapshots: 7 days
- AOF files: 3 days

**Backup Method:**
```bash
# RDB snapshot backup
redis-cli --rdb /backups/redis/dump_$(date +%Y%m%d_%H%M%S).rdb

# Or configure automatic snapshots in redis.conf
# save 21600 1  # Save after 6 hours if at least 1 key changed
```

**Redis Configuration:**
```conf
# redis.conf
save 21600 1
dir /backups/redis
dbfilename dump.rdb

# Enable AOF for better durability
appendonly yes
appendfsync everysec
```

**Automated Backup Script:**
```bash
#!/bin/bash
# scripts/backup-redis.sh

BACKUP_DIR="/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Trigger RDB snapshot
redis-cli BGSAVE

# Wait for snapshot to complete
while [ "$(redis-cli LASTSAVE)" = "$(redis-cli LASTSAVE)" ]; do
  sleep 1
done

# Copy snapshot
cp /var/lib/redis/dump.rdb $BACKUP_DIR/dump_${DATE}.rdb

# Upload to S3 (if configured)
if [ -n "$S3_BACKUP_BUCKET" ]; then
  aws s3 cp $BACKUP_DIR/dump_${DATE}.rdb \
    s3://$S3_BACKUP_BUCKET/redis/dump_${DATE}.rdb
fi

# Cleanup old backups
find $BACKUP_DIR -name "dump_*.rdb" -mtime +7 -delete
```

## Cloud Storage (AWS S3)

**S3 Bucket Configuration:**
- **Bucket Name**: `phishing-detection-backups-{environment}`
- **Region**: Same as primary infrastructure
- **Storage Class**: Standard for recent backups, Glacier for older backups
- **Lifecycle Policy**:
  - Move to Glacier after 30 days
  - Delete after 12 months

**S3 Bucket Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::ACCOUNT_ID:role/BackupRole"
      },
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::phishing-detection-backups-*/*"
    }
  ]
}
```

## Recovery Procedures

### PostgreSQL Recovery

**Point-in-Time Recovery:**
```bash
# Restore from backup
pg_restore -h localhost -U postgres -d phishing_detection \
  --clean \
  --if-exists \
  /backups/postgres/full_YYYYMMDD_HHMMSS.dump

# Point-in-time recovery using WAL
# 1. Restore base backup
pg_basebackup -h primary-server -D /var/lib/postgresql/data -P

# 2. Create recovery.conf
echo "restore_command = 'cp /backups/postgres/wal/%f %p'" > recovery.conf
echo "recovery_target_time = '2024-01-01 12:00:00'" >> recovery.conf

# 3. Start PostgreSQL
```

### MongoDB Recovery

**Full Restore:**
```bash
# Restore from archive
mongorestore --uri="mongodb://localhost:27017/phishing_detection" \
  --archive=/backups/mongodb/full_YYYYMMDD_HHMMSS.archive \
  --gzip
```

**Point-in-Time Recovery:**
```bash
# Restore base backup
mongorestore --uri="mongodb://localhost:27017/phishing_detection" \
  --archive=/backups/mongodb/full_YYYYMMDD_HHMMSS.archive \
  --gzip

# Replay oplog to specific timestamp
mongorestore --uri="mongodb://localhost:27017/phishing_detection" \
  --archive=/backups/mongodb/oplog_YYYYMMDD_HHMMSS.archive \
  --gzip \
  --oplogReplay \
  --oplogLimit 1234567890
```

### Redis Recovery

**RDB Restore:**
```bash
# Stop Redis
redis-cli SHUTDOWN

# Copy backup file
cp /backups/redis/dump_YYYYMMDD_HHMMSS.rdb /var/lib/redis/dump.rdb

# Start Redis
redis-server /etc/redis/redis.conf
```

## Monitoring and Alerts

**Backup Monitoring:**
- Monitor backup job completion status
- Alert on backup failures
- Track backup sizes and durations
- Verify backup integrity

**Monitoring Script:**
```bash
#!/bin/bash
# scripts/verify-backups.sh

# Check PostgreSQL backup
if [ -f /backups/postgres/full_$(date +%Y%m%d)*.dump ]; then
  echo "✓ PostgreSQL backup exists"
else
  echo "✗ PostgreSQL backup missing!"
  # Send alert
fi

# Check MongoDB backup
if [ -f /backups/mongodb/full_$(date +%Y%m%d)*.archive ]; then
  echo "✓ MongoDB backup exists"
else
  echo "✗ MongoDB backup missing!"
  # Send alert
fi

# Check Redis backup
if [ -f /backups/redis/dump_$(date +%Y%m%d)*.rdb ]; then
  echo "✓ Redis backup exists"
else
  echo "✗ Redis backup missing!"
  # Send alert
fi
```

## Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: 4 hours
2. **RPO (Recovery Point Objective)**: 1 hour

**Recovery Steps:**
1. Identify the point of failure
2. Restore from most recent backup
3. Apply transaction logs/oplog to point-in-time
4. Verify data integrity
5. Resume operations

## Testing

**Backup Testing Schedule:**
- Monthly: Test restore procedures
- Quarterly: Full disaster recovery drill
- Annually: Review and update backup strategy

**Test Restore Procedure:**
```bash
# Test PostgreSQL restore on test server
pg_restore -h test-server -U postgres -d phishing_detection_test \
  --clean \
  /backups/postgres/full_YYYYMMDD_HHMMSS.dump

# Verify data integrity
psql -h test-server -U postgres -d phishing_detection_test \
  -c "SELECT COUNT(*) FROM threats;"
```

## Automation

**Cron Jobs:**
```cron
# PostgreSQL daily backup at 2 AM
0 2 * * * /scripts/backup-postgres.sh

# MongoDB daily backup at 3 AM
0 3 * * * /scripts/backup-mongodb.sh

# Redis backup every 6 hours
0 */6 * * * /scripts/backup-redis.sh

# Backup verification daily at 4 AM
0 4 * * * /scripts/verify-backups.sh
```

## Security

- Encrypt backups at rest (S3 server-side encryption)
- Use IAM roles for S3 access
- Secure backup storage locations
- Regular access audits
- Backup encryption keys rotation

## Compliance

- GDPR: Ensure backups don't contain PII beyond retention period
- SOC 2: Maintain audit trail of backup operations
- HIPAA (if applicable): Encrypt all backups containing PHI
