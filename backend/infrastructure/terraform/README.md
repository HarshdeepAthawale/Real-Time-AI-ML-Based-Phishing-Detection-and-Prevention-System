# Infrastructure as Code - Terraform

This directory contains Terraform configurations for provisioning AWS infrastructure.

## Structure

```
terraform/
├── modules/          # Reusable Terraform modules
│   ├── vpc/         # VPC and networking
│   ├── rds/         # PostgreSQL database
│   ├── redis/       # ElastiCache Redis
│   ├── s3/          # S3 buckets
│   └── ecs/         # ECS cluster and services
├── environments/    # Environment-specific variables
├── main.tf          # Main configuration
├── variables.tf     # Variable definitions
└── outputs.tf       # Output values
```

## Prerequisites

1. **AWS CLI configured:**
   ```bash
   aws configure
   ```

2. **Terraform installed:**
   ```bash
   terraform version
   ```

3. **S3 bucket for Terraform state:**
   - Create an S3 bucket for storing Terraform state
   - Update `main.tf` backend configuration

## Usage

### Initialize Terraform

```bash
terraform init
```

### Plan Infrastructure

```bash
# Development
terraform plan -var-file=environments/dev.tfvars

# Production
terraform plan -var-file=environments/prod.tfvars
```

### Apply Infrastructure

```bash
# Development
terraform apply -var-file=environments/dev.tfvars

# Production (requires confirmation)
terraform apply -var-file=environments/prod.tfvars
```

### Destroy Infrastructure

```bash
terraform destroy -var-file=environments/dev.tfvars
```

## Environment Variables

Create `environments/dev.tfvars` and `environments/prod.tfvars` from the example files:

```bash
cp environments/dev.tfvars.example environments/dev.tfvars
cp environments/prod.tfvars.example environments/prod.tfvars
```

**Important:** Never commit `.tfvars` files with real credentials to version control.

## Modules

### VPC Module

Creates:
- VPC with public and private subnets
- Internet Gateway and NAT Gateways
- Security groups for each service tier
- VPC endpoints for S3

### RDS Module

Creates:
- PostgreSQL 15.x database
- Multi-AZ support for production
- Automated backups
- Encryption at rest

### Redis Module

Creates:
- ElastiCache Redis cluster
- Encryption in-transit and at-rest
- Automatic failover for production

### S3 Module

Creates:
- Models bucket
- Training data bucket
- Logs bucket
- Artifacts bucket

All buckets have:
- Versioning enabled
- Encryption enabled
- Lifecycle policies

### ECS Module

Creates:
- ECS Fargate cluster
- Application Load Balancer
- CloudWatch log groups
- IAM roles for tasks
- Service discovery namespace

## Outputs

After applying, Terraform outputs:
- VPC ID and subnet IDs
- Database endpoint
- Redis endpoint
- S3 bucket names
- ECS cluster name
- ALB DNS name

View outputs:
```bash
terraform output
```

## Security Considerations

1. **Secrets Management:**
   - Use AWS Secrets Manager for database passwords
   - Never commit secrets to version control

2. **Network Security:**
   - All services run in private subnets
   - Only API Gateway is exposed via ALB
   - Security groups restrict traffic

3. **Encryption:**
   - RDS encryption at rest
   - Redis encryption in-transit and at-rest
   - S3 encryption enabled

## Cost Optimization

- Use smaller instance types for development
- Enable multi-AZ only for production
- Use lifecycle policies for S3
- Consider Reserved Instances for production

## Troubleshooting

### Terraform state locked

```bash
# If state is locked, check for running operations
# Or force unlock (use with caution)
terraform force-unlock <LOCK_ID>
```

### Module not found

```bash
# Re-initialize Terraform
terraform init -upgrade
```

### Variable not set

Ensure all required variables are set in `.tfvars` file or as environment variables prefixed with `TF_VAR_`.

## Best Practices

1. Always run `terraform plan` before `apply`
2. Use version control for Terraform code
3. Store state remotely (S3 backend)
4. Use workspaces for multiple environments
5. Review changes carefully before applying
6. Use `terraform fmt` to format code
7. Use `terraform validate` to check syntax
