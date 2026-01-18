# Main Terraform Configuration
# This file orchestrates all infrastructure modules

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    # Configure backend in terraform/backend.tfvars
    # bucket = "phishing-detection-terraform-state"
    # key    = "terraform.tfstate"
    # region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "db_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  environment       = var.environment
  vpc_cidr          = "10.0.0.0/16"
  availability_zones = var.availability_zones
}

# RDS Module
module "rds" {
  source = "./modules/rds"

  environment        = var.environment
  subnet_ids         = module.vpc.private_subnet_ids
  security_group_id  = module.vpc.rds_security_group_id
  db_instance_class  = var.db_instance_class
  db_password        = var.db_password
  multi_az          = var.environment == "prod"
}

# Redis Module
module "redis" {
  source = "./modules/redis"

  environment      = var.environment
  subnet_ids       = module.vpc.private_subnet_ids
  security_group_id = module.vpc.redis_security_group_id
  node_type        = var.redis_node_type
}

# S3 Module
module "s3" {
  source = "./modules/s3"

  environment = var.environment
}

# ECS Module
module "ecs" {
  source = "./modules/ecs"

  environment       = var.environment
  vpc_id            = module.vpc.vpc_id
  subnet_ids        = module.vpc.private_subnet_ids
  security_group_id = module.vpc.ecs_services_security_group_id
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "db_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.db_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.redis.redis_endpoint
}

output "s3_buckets" {
  description = "S3 bucket names"
  value = {
    models    = module.s3.models_bucket
    training  = module.s3.training_bucket
    logs      = module.s3.logs_bucket
    artifacts = module.s3.artifacts_bucket
  }
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = module.ecs.cluster_name
}

output "alb_dns_name" {
  description = "Application Load Balancer DNS name"
  value       = module.ecs.alb_dns_name
}
