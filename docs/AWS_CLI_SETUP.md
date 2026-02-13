# AWS Setup via CLI

This guide configures the full AWS infrastructure for the Phishing Detection System using CLI commands only.

## Prerequisites

- **AWS CLI** v2: `pip install awscli` or [install from AWS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- **Terraform** >= 1.5: [Install Terraform](https://developer.hashicorp.com/terraform/install)
- **AWS credentials** configured: `aws configure`

## Quick Start

**Interactive (recommended):**
```bash
# Prompts for AWS keys (if needed), DB password, API_KEY_SECRET, JWT_SECRET
./scripts/configure-credentials.sh
source .env.credentials
./scripts/aws-setup.sh all
```

**Manual:**
```bash
# 1. aws configure   # Enter your Access Key ID and Secret
# 2. Set required vars
export TF_VAR_db_password="your-secure-password"
export SSM_API_KEY_SECRET="your-api-key-secret"
export SSM_JWT_SECRET="your-jwt-secret"

# 3. Run full setup
./scripts/aws-setup.sh all
```

## Step-by-Step

### 0. Verify Prerequisites

```bash
./scripts/aws-setup.sh check
```

### 1. Create IAM User (if needed)

Create a dedicated IAM user for Terraform and CI/CD:

```bash
# Create user
aws iam create-user --user-name phishing-detection-deployer

# Create and attach policy (use scripts/aws-iam-policy.json)
aws iam put-user-policy \
  --user-name phishing-detection-deployer \
  --policy-name PhishingDetectionDeploy \
  --policy-document file://scripts/aws-iam-policy.json

# Create access key
aws iam create-access-key --user-name phishing-detection-deployer
# Store AccessKeyId and SecretAccessKey securely
```

For production, prefer an IAM role with broader Terraform permissions (EC2, RDS, ECS, VPC, S3, etc.); the policy above covers CI/CD and ECR. Terraform needs full permissions for the resources it creates.

### 2. Bootstrap Terraform Backend

Creates S3 bucket and DynamoDB table for Terraform state:

```bash
./scripts/aws-setup.sh bootstrap
```

Or manually:

```bash
export AWS_REGION=ap-south-1
aws s3 mb s3://phishing-detection-terraform-state-1768755350 --region $AWS_REGION
aws s3api put-bucket-versioning \
  --bucket phishing-detection-terraform-state-1768755350 \
  --versioning-configuration Status=Enabled
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region $AWS_REGION
```

### 3. Create ECR Repositories

```bash
./scripts/aws-setup.sh ecr
```

Creates 9 repositories: `api-gateway`, `detection-api`, `threat-intel`, `extension-api`, `sandbox-service`, `learning-pipeline`, `nlp-service`, `url-service`, `visual-service`.

### 4. Run Terraform

```bash
export TF_VAR_db_password="your-secure-password"
export ENVIRONMENT=dev   # or prod
./scripts/aws-setup.sh terraform
```

Terraform provisions:

- **VPC** – Public/private subnets, NAT gateways, security groups
- **RDS** – PostgreSQL 16 (phishing_detection DB)
- **ElastiCache** – Redis
- **S3** – Models, training data, logs, artifacts buckets
- **ECS** – Fargate cluster, ALB, 8+ microservices
- **CloudWatch** – Log groups

### 5. Store Secrets in SSM

```bash
export SSM_API_KEY_SECRET="your-api-key-secret"
export SSM_JWT_SECRET="your-jwt-secret"
./scripts/aws-setup.sh secrets
```

Or manually:

```bash
aws ssm put-parameter --name "/phishing-detection/dev/API_KEY_SECRET" \
  --value "your-value" --type SecureString --overwrite --region ap-south-1
aws ssm put-parameter --name "/phishing-detection/dev/JWT_SECRET" \
  --value "your-value" --type SecureString --overwrite --region ap-south-1
```

### 6. Upload ML Models to S3

```bash
./scripts/aws-setup.sh models
```

Syncs local models from `backend/ml-services/*/models/` to `s3://phishing-detection-models-{env}/`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region | `ap-south-1` |
| `ENVIRONMENT` | Environment (dev/staging/prod) | `dev` |
| `TF_VAR_db_password` | RDS master password | *required* |
| `TF_STATE_BUCKET` | Terraform state bucket | `phishing-detection-terraform-state-1768755350` |
| `SSM_API_KEY_SECRET` | For SSM secrets step | optional |
| `SSM_JWT_SECRET` | For SSM secrets step | optional |

## Production

For production, use a separate tfvars file:

```bash
cp backend/infrastructure/terraform/environments/prod.tfvars.example \
   backend/infrastructure/terraform/environments/prod.tfvars
# Edit prod.tfvars: set certificate_arn for HTTPS, stronger db/redis instance types
```

Then:

```bash
export ENVIRONMENT=prod
export TF_VAR_db_password="strong-production-password"
./scripts/aws-setup.sh all
```

**SSL**: Request an ACM certificate for your domain, then set `certificate_arn` in `prod.tfvars`.

## Post-Setup

1. **Get ALB DNS**: `cd backend/infrastructure/terraform && terraform output alb_dns_name`
2. **Push Docker images**: Run CI/CD pipeline or `docker build` + `docker push` to ECR
3. **Create API key**: Run `backend/shared/scripts/create-initial-setup.ts` and store in GitHub secrets as `TEST_API_KEY`
4. **Configure GitHub Secrets**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ACCOUNT_ID`, `TF_VAR_DB_PASSWORD`, `TEST_API_KEY`

**Bucket name taken?** If S3 returns `BucketAlreadyExists`, the name is taken globally. Use a unique name:
`export TF_STATE_BUCKET=phishing-detection-tfstate-$(aws sts get-caller-identity --query AccountId --output text)`
Then update `backend/infrastructure/terraform/backend.tfvars` and the `backend "s3"` block in `main.tf` to match.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `aws: command not found` | Install AWS CLI v2 |
| `An error occurred (InvalidClientTokenId)` | Run `aws configure` with valid credentials |
| `Error acquiring the state lock` | Wait for other Terraform run, or force-unlock: `terraform force-unlock LOCK_ID` |
| ECS tasks fail to start | Ensure ECR images exist (`:latest`); run CI pipeline to build/push |
| RDS connection refused | Check security groups allow ECS → RDS on 5432 |
