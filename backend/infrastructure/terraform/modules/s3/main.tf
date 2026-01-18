# S3 Buckets Module

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default     = {}
}

locals {
  bucket_names = {
    models     = "phishing-detection-models-${var.environment}"
    training   = "phishing-detection-training-data-${var.environment}"
    logs       = "phishing-detection-logs-${var.environment}"
    artifacts  = "phishing-detection-artifacts-${var.environment}"
  }
}

# Models Bucket
resource "aws_s3_bucket" "models" {
  bucket = local.bucket_names.models

  tags = merge(
    var.tags,
    {
      Name = local.bucket_names.models
      Type = "models"
    }
  )
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    id     = "delete-old-versions"
    status = "Enabled"

    filter {
      prefix = ""
    }

    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}

# Training Data Bucket
resource "aws_s3_bucket" "training" {
  bucket = local.bucket_names.training

  tags = merge(
    var.tags,
    {
      Name = local.bucket_names.training
      Type = "training"
    }
  )
}

resource "aws_s3_bucket_versioning" "training" {
  bucket = aws_s3_bucket.training.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "training" {
  bucket = aws_s3_bucket.training.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Logs Bucket
resource "aws_s3_bucket" "logs" {
  bucket = local.bucket_names.logs

  tags = merge(
    var.tags,
    {
      Name = local.bucket_names.logs
      Type = "logs"
    }
  )
}

resource "aws_s3_bucket_versioning" "logs" {
  bucket = aws_s3_bucket.logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"

    filter {
      prefix = ""
    }

    transition {
      days          = 30
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# Artifacts Bucket
resource "aws_s3_bucket" "artifacts" {
  bucket = local.bucket_names.artifacts

  tags = merge(
    var.tags,
    {
      Name = local.bucket_names.artifacts
      Type = "artifacts"
    }
  )
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "delete-old-versions"
    status = "Enabled"

    filter {
      prefix = ""
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# Outputs
output "models_bucket" {
  description = "Models bucket name"
  value       = aws_s3_bucket.models.id
}

output "training_bucket" {
  description = "Training data bucket name"
  value       = aws_s3_bucket.training.id
}

output "logs_bucket" {
  description = "Logs bucket name"
  value       = aws_s3_bucket.logs.id
}

output "artifacts_bucket" {
  description = "Artifacts bucket name"
  value       = aws_s3_bucket.artifacts.id
}
