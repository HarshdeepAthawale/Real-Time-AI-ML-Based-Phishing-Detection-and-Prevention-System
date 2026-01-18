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
    bucket         = "phishing-detection-terraform-state-1768755350"
    key            = "terraform.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
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

# VPC Module
module "vpc" {
  source = "./modules/vpc"

  environment        = var.environment
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = var.availability_zones
}

# RDS Module
module "rds" {
  source = "./modules/rds"

  environment       = var.environment
  subnet_ids        = module.vpc.private_subnet_ids
  security_group_id = module.vpc.rds_security_group_id
  db_instance_class = var.db_instance_class
  db_password       = var.db_password
  multi_az          = var.environment == "prod"
}

# Redis Module
module "redis" {
  source = "./modules/redis"

  environment       = var.environment
  subnet_ids        = module.vpc.private_subnet_ids
  security_group_id = module.vpc.redis_security_group_id
  node_type         = var.redis_node_type
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
