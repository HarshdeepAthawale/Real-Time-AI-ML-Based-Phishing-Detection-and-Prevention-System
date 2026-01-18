# Phase 1 Infrastructure Setup Guide

This guide will help you set up the Phase 1 infrastructure for the Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System.

## Prerequisites Checklist

- [ ] AWS Account with appropriate permissions
- [ ] Docker Desktop installed and running
- [ ] Terraform CLI installed (`terraform version`)
- [ ] Node.js 20+ installed (`node --version`)
- [ ] Python 3.11+ installed (`python --version`)
- [ ] AWS CLI configured (`aws configure`)
- [ ] GitHub account for CI/CD (if using GitHub Actions)

## Step-by-Step Setup

### 1. Environment Configuration

```bash
cd backend
cp env.template .env
# Edit .env with your actual values
```

### 2. Local Development Setup

#### Start Infrastructure Services

```bash
# Start PostgreSQL, MongoDB, and Redis
docker-compose up -d postgres mongodb redis

# Verify services are running
docker-compose ps
```

#### Setup API Gateway

```bash
cd api-gateway
npm install
npm run build
npm run dev
```

The API Gateway should now be running at `http://localhost:3000`

#### Test Health Endpoint

```bash
curl http://localhost:3000/health
```

Expected response:
```json
{
  "status": "ok",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### 3. AWS Infrastructure Setup

#### Configure Terraform Backend

1. Create an S3 bucket for Terraform state:
   ```bash
   aws s3 mb s3://phishing-detection-terraform-state
   ```

2. Enable versioning:
   ```bash
   aws s3api put-bucket-versioning \
     --bucket phishing-detection-terraform-state \
     --versioning-configuration Status=Enabled
   ```

3. Update `infrastructure/terraform/main.tf` backend configuration:
   ```hcl
   backend "s3" {
     bucket = "phishing-detection-terraform-state"
     key    = "terraform.tfstate"
     region = "us-east-1"
   }
   ```

#### Configure Environment Variables

```bash
cd infrastructure/terraform
cp environments/dev.tfvars.example environments/dev.tfvars
# Edit dev.tfvars with your values
```

**Important:** Set a strong database password in `dev.tfvars`:
```hcl
db_password = "YourStrongPasswordHere123!"
```

#### Initialize Terraform

```bash
terraform init
```

#### Plan Infrastructure

```bash
terraform plan -var-file=environments/dev.tfvars
```

Review the plan carefully. You should see:
- VPC and networking resources
- RDS PostgreSQL instance
- ElastiCache Redis cluster
- S3 buckets
- ECS cluster

#### Apply Infrastructure

```bash
terraform apply -var-file=environments/dev.tfvars
```

Type `yes` when prompted. This will take 10-15 minutes.

#### Save Outputs

After applying, save the outputs:
```bash
terraform output > terraform-outputs.txt
```

### 4. CI/CD Setup

#### GitHub Secrets Configuration

Configure the following secrets in your GitHub repository:

1. Go to Settings → Secrets and variables → Actions
2. Add the following secrets:

   - `AWS_ACCESS_KEY_ID` - Your AWS access key
   - `AWS_SECRET_ACCESS_KEY` - Your AWS secret key
   - `AWS_ACCOUNT_ID` - Your AWS account ID
   - `TF_VAR_DB_PASSWORD` - Database password for dev
   - `TF_VAR_DB_PASSWORD_PROD` - Database password for prod

#### Test CI/CD Pipeline

1. Push changes to `develop` branch to trigger dev deployment
2. Check GitHub Actions tab for pipeline status
3. Verify deployment in AWS Console

### 5. Verification

#### Verify Infrastructure

```bash
# Check VPC
aws ec2 describe-vpcs --filters "Name=tag:Name,Values=phishing-detection-vpc-dev"

# Check RDS
aws rds describe-db-instances --db-instance-identifier phishing-detection-db-dev

# Check Redis
aws elasticache describe-replication-groups --replication-group-id phishing-detection-redis-dev

# Check S3 buckets
aws s3 ls | grep phishing-detection

# Check ECS cluster
aws ecs describe-clusters --clusters phishing-detection-cluster-dev
```

#### Verify Services

```bash
# Test API Gateway health
curl http://localhost:3000/health

# Test database connection (if configured)
psql $DATABASE_URL -c "SELECT version();"

# Test Redis connection
redis-cli -u $REDIS_URL ping
```

## Troubleshooting

### Docker Issues

**Problem:** Services won't start
```bash
# Check Docker is running
docker ps

# Check logs
docker-compose logs

# Restart services
docker-compose restart
```

### Terraform Issues

**Problem:** State locked
```bash
# Check for running operations
# If needed, force unlock (use with caution)
terraform force-unlock <LOCK_ID>
```

**Problem:** Module not found
```bash
# Re-initialize
terraform init -upgrade
```

### AWS Issues

**Problem:** Insufficient permissions
- Verify IAM user/role has required permissions
- Check CloudTrail for denied actions

**Problem:** Resource limits
- Check AWS service quotas
- Request limit increases if needed

## Next Steps

After completing Phase 1:

1. ✅ Verify all infrastructure is running
2. ✅ Test service-to-service communication
3. ✅ Set up monitoring dashboards
4. ✅ Document deployment procedures
5. ➡️ Proceed to Phase 2: Database Schema & Data Models

## Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Terraform and Docker logs
3. Consult AWS CloudWatch logs
4. Refer to project documentation
