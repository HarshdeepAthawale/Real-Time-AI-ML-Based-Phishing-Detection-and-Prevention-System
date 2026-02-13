# Terraform Backend Bootstrap

Before running `terraform init` or `terraform plan`, you must create the Terraform backend resources in AWS.

## One-Time Manual Steps

### 1. Create S3 Bucket for Terraform State

The Terraform state is stored in S3 with encryption. Create the bucket in your target region:

```bash
aws s3 mb s3://phishing-detection-terraform-state-1768755350 --region ap-south-1
aws s3api put-bucket-versioning \
  --bucket phishing-detection-terraform-state-1768755350 \
  --versioning-configuration Status=Enabled
```

### 2. Create DynamoDB Table for State Locking

Terraform uses DynamoDB for state locking to prevent concurrent modifications:

```bash
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region ap-south-1
```

### 3. Configure AWS Credentials

Ensure one of the following is set:

- **Environment variables:** `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- **AWS CLI profile:** `aws configure` or `export AWS_PROFILE=your-profile`
- **IAM role:** When running from EC2/ECS/Lambda with an attached role

### 4. Database Password (Sensitive)

**Do not commit `db_password` to version control.** Use one of these approaches:

**Option A: Environment variable**
```bash
export TF_VAR_db_password="your-secure-password"
terraform plan -var-file=environments/dev.tfvars
```

**Option B: Copy and customize dev.tfvars (add to .gitignore)**
```bash
cp environments/dev.tfvars.example environments/dev.tfvars
# Edit dev.tfvars and set db_password - ensure dev.tfvars is in .gitignore
terraform plan -var-file=environments/dev.tfvars
```

**Option C: Command-line variable (avoid for scripts)**
```bash
terraform plan -var-file=environments/dev.tfvars -var="db_password=your-secure-password"
```

### 5. Production (prod.tfvars)

For production, copy `environments/prod.tfvars.example` to `environments/prod.tfvars` and customize. Never commit `db_password`; use `TF_VAR_db_password` from GitHub secret `TF_VAR_DB_PASSWORD_PROD`.

### 6. Initialize Terraform

```bash
cd backend/infrastructure/terraform
terraform init
terraform validate
```

For first-time init with backend config file:
```bash
terraform init -backend-config=backend.tfvars
```

## S3 Bucket Names (Terraform Outputs)

When Terraform is applied, it creates these S3 buckets (for `dev` environment):

| Purpose    | Bucket Name                                |
|-----------|--------------------------------------------|
| Models    | `phishing-detection-models-dev`            |
| Training  | `phishing-detection-training-data-dev`     |
| Logs      | `phishing-detection-logs-dev`              |
| Artifacts | `phishing-detection-artifacts-dev`         |

Set these in your application `.env` to match when deploying to AWS.
