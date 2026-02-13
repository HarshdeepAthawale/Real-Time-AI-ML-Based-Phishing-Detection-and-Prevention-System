#!/usr/bin/env bash
# =============================================================================
# AWS Setup via CLI - Phishing Detection System
# Configures the full AWS infrastructure: bootstrap, ECR, Terraform, secrets
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/backend/infrastructure/terraform"
REGION="${AWS_REGION:-ap-south-1}"
ENV="${ENVIRONMENT:-dev}"
# Must match backend.tfvars + main.tf. Override TF_STATE_BUCKET if name is taken globally.
TF_STATE_BUCKET="${TF_STATE_BUCKET:-phishing-detection-terraform-state-047385030558}"
TF_LOCK_TABLE="${TF_LOCK_TABLE:-terraform-state-lock}"

# ECR repository names (must match CI/CD and ECS task definitions)
ECR_REPOS=(
  phishing-detection-api-gateway
  phishing-detection-detection-api
  phishing-detection-threat-intel
  phishing-detection-extension-api
  phishing-detection-sandbox-service
  phishing-detection-learning-pipeline
  phishing-detection-nlp-service
  phishing-detection-url-service
  phishing-detection-visual-service
)

log() { echo "[$(date +%H:%M:%S)] $*"; }
err() { echo "[ERROR] $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 0. Prerequisites
# -----------------------------------------------------------------------------
check_prereqs() {
  log "Checking prerequisites..."
  command -v aws >/dev/null 2>&1 || err "aws CLI not installed. Run: pip install awscli or install AWS CLI v2"
  command -v terraform >/dev/null 2>&1 || err "terraform not installed. See https://developer.hashicorp.com/terraform/install"
  aws sts get-caller-identity >/dev/null 2>&1 || err "AWS credentials not configured. Run: aws configure"
  log "Prerequisites OK (AWS CLI, Terraform, credentials)"
}

# -----------------------------------------------------------------------------
# 1. Bootstrap - S3 + DynamoDB (Terraform backend)
# -----------------------------------------------------------------------------
bootstrap_terraform_backend() {
  log "Bootstrap: Terraform backend (S3 + DynamoDB)..."
  
  if aws s3api head-bucket --bucket "$TF_STATE_BUCKET" 2>/dev/null; then
    log "S3 bucket $TF_STATE_BUCKET already exists"
  else
    if ! aws s3 mb "s3://$TF_STATE_BUCKET" --region "$REGION"; then
      err "S3 bucket '$TF_STATE_BUCKET' creation failed. Name may be taken globally. Set TF_STATE_BUCKET to a unique name (e.g. phishing-detection-tfstate-\$ACCOUNT_ID)"
    fi
    aws s3api put-bucket-versioning \
      --bucket "$TF_STATE_BUCKET" \
      --versioning-configuration Status=Enabled
    aws s3api put-bucket-encryption \
      --bucket "$TF_STATE_BUCKET" \
      --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
    log "Created S3 bucket: $TF_STATE_BUCKET"
  fi

  if aws dynamodb describe-table --table-name "$TF_LOCK_TABLE" --region "$REGION" 2>/dev/null; then
    log "DynamoDB table $TF_LOCK_TABLE already exists"
  else
    aws dynamodb create-table \
      --table-name "$TF_LOCK_TABLE" \
      --attribute-definitions AttributeName=LockID,AttributeType=S \
      --key-schema AttributeName=LockID,KeyType=HASH \
      --billing-mode PAY_PER_REQUEST \
      --region "$REGION"
    log "Created DynamoDB table: $TF_LOCK_TABLE"
  fi
}

# -----------------------------------------------------------------------------
# 2. ECR Repositories
# -----------------------------------------------------------------------------
create_ecr_repos() {
  log "Creating ECR repositories..."
  for repo in "${ECR_REPOS[@]}"; do
    if aws ecr describe-repositories --repository-names "$repo" --region "$REGION" 2>/dev/null; then
      log "ECR $repo already exists"
    else
      aws ecr create-repository \
        --repository-name "$repo" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 \
        --region "$REGION"
      log "Created ECR: $repo"
    fi
  done
}

# -----------------------------------------------------------------------------
# 3. Terraform init, plan, apply
# -----------------------------------------------------------------------------
run_terraform() {
  log "Running Terraform (env=$ENV)..."
  cd "$TERRAFORM_DIR"

  export TF_VAR_db_password="${TF_VAR_db_password:-}"
  if [[ -z "$TF_VAR_db_password" && -f "environments/${ENV}.tfvars" ]]; then
    # Try to read from tfvars if not set (dev.tfvars has default - avoid for prod!)
    if [[ "$ENV" == "prod" ]]; then
      err "TF_VAR_db_password must be set for prod. Example: export TF_VAR_db_password='your-secure-password'"
    fi
  fi

  terraform init -backend-config="backend.tfvars" -reconfigure
  terraform validate

  VAR_FILE="environments/${ENV}.tfvars"
  [[ -f "$VAR_FILE" ]] || err "Var file not found: $VAR_FILE"

  terraform plan -var-file="$VAR_FILE" -out=tfplan
  terraform apply -auto-approve tfplan
  terraform output -json > terraform-outputs.json 2>/dev/null || true
  log "Terraform apply completed"
}

# -----------------------------------------------------------------------------
# 4. Post-provision: SSM Parameters (secrets)
# -----------------------------------------------------------------------------
setup_ssm_secrets() {
  log "Setting up SSM Parameters (secrets)..."
  PREFIX="/phishing-detection/$ENV"
  # Only create if values are provided via env
  for key in API_KEY_SECRET JWT_SECRET; do
    val="SSM_${key}"
    val="${!val:-}"
    if [[ -n "$val" ]]; then
      aws ssm put-parameter \
        --name "$PREFIX/$key" \
        --value "$val" \
        --type SecureString \
        --overwrite \
        --region "$REGION" 2>/dev/null && log "Set SSM $PREFIX/$key" || log "Skip $key (already exists or no permission)"
    fi
  done
}

# -----------------------------------------------------------------------------
# 5. Upload models to S3 (if local models exist)
# -----------------------------------------------------------------------------
upload_models() {
  MODELS_BUCKET="phishing-detection-models-$ENV"
  MODELS_SRC="$PROJECT_ROOT/backend/ml-services"
  if aws s3 ls "s3://$MODELS_BUCKET" --region "$REGION" 2>/dev/null; then
    for svc in nlp-service visual-service; do
      MODEL_PATH="$MODELS_SRC/$svc/models"
      if [[ -d "$MODEL_PATH" ]]; then
        log "Uploading models from $svc..."
        aws s3 sync "$MODEL_PATH" "s3://$MODELS_BUCKET/models/$svc/" --region "$REGION" || true
      fi
    done
  fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
  echo "=============================================="
  echo " AWS Setup - Phishing Detection System"
  echo " Region: $REGION | Env: $ENV"
  echo "=============================================="

  PHASE="${1:-all}"
  case "$PHASE" in
    check)      check_prereqs ;;
    bootstrap)   check_prereqs; bootstrap_terraform_backend ;;
    ecr)         check_prereqs; create_ecr_repos ;;
    terraform)   check_prereqs; run_terraform ;;
    secrets)     check_prereqs; setup_ssm_secrets ;;
    models)      check_prereqs; upload_models ;;
    all)
      check_prereqs
      bootstrap_terraform_backend
      create_ecr_repos
      run_terraform
      setup_ssm_secrets
      upload_models
      log "AWS setup complete. ALB DNS: terraform output alb_dns_name"
      ;;
    *)
      echo "Usage: $0 {check|bootstrap|ecr|terraform|secrets|models|all}"
      echo ""
      echo "  check     - Verify AWS CLI, Terraform, credentials"
      echo "  bootstrap - Create S3 bucket + DynamoDB for Terraform state"
      echo "  ecr       - Create ECR repositories for Docker images"
      echo "  terraform - Run terraform init, plan, apply"
      echo "  secrets   - Store API_KEY_SECRET, JWT_SECRET in SSM (set SSM_* env vars)"
      echo "  models    - Sync local ML models to S3"
      echo "  all       - Run bootstrap + ecr + terraform + secrets + models"
      echo ""
      echo "Env vars: AWS_REGION, ENVIRONMENT, TF_VAR_db_password, SSM_API_KEY_SECRET, SSM_JWT_SECRET"
      exit 1
      ;;
  esac
}

main "$@"
