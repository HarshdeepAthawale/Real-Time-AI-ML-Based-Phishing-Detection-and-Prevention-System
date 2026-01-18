# ECS Fargate Cluster Module

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs"
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group ID for ECS tasks"
  type        = string
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS (optional)"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default     = {}
}

# CloudWatch Log Groups for each service
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/phishing-detection/api-gateway-${var.environment}"
  retention_in_days = var.environment == "prod" ? 30 : 7
  tags              = var.tags
}

resource "aws_cloudwatch_log_group" "detection_api" {
  name              = "/phishing-detection/detection-api-${var.environment}"
  retention_in_days = var.environment == "prod" ? 30 : 7
  tags              = var.tags
}

resource "aws_cloudwatch_log_group" "ml_services" {
  name              = "/phishing-detection/ml-services-${var.environment}"
  retention_in_days = var.environment == "prod" ? 30 : 7
  tags              = var.tags
}

resource "aws_cloudwatch_log_group" "threat_intel" {
  name              = "/phishing-detection/threat-intel-${var.environment}"
  retention_in_days = var.environment == "prod" ? 30 : 7
  tags              = var.tags
}

# General ECS log group (for other services)
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/phishing-detection-${var.environment}"
  retention_in_days = var.environment == "prod" ? 30 : 7
  tags              = var.tags
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_task_execution" {
  name = "phishing-detection-ecs-task-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM Role for ECS Tasks (application role)
resource "aws_iam_role" "ecs_task" {
  name = "phishing-detection-ecs-task-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# Policy for S3 access
resource "aws_iam_role_policy" "s3_access" {
  name = "s3-access-${var.environment}"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::phishing-detection-*",
          "arn:aws:s3:::phishing-detection-*/*"
        ]
      }
    ]
  })
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "phishing-detection-cluster-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = var.tags
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "phishing-detection-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [var.security_group_id]
  subnets            = var.subnet_ids

  enable_deletion_protection = var.environment == "prod"

  tags = var.tags
}

# Target Group for API Gateway
resource "aws_lb_target_group" "api_gateway" {
  name        = "pd-api-gw-${var.environment}"
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    path                = "/health"
    protocol            = "HTTP"
    matcher             = "200"
  }

  tags = var.tags
}

# ALB Listener - Use HTTP for dev, HTTPS for prod
resource "aws_lb_listener" "main" {
  load_balancer_arn = aws_lb.main.arn
  port              = var.environment == "prod" ? "443" : "80"
  protocol          = var.environment == "prod" ? "HTTPS" : "HTTP"
  ssl_policy        = var.environment == "prod" ? "ELBSecurityPolicy-TLS13-1-2-2021-06" : null
  certificate_arn   = var.environment == "prod" && var.certificate_arn != "" ? var.certificate_arn : null

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api_gateway.arn
  }
}

# HTTP to HTTPS redirect (only for prod)
resource "aws_lb_listener" "http_redirect" {
  count             = var.environment == "prod" ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# Service Discovery Namespace
# Note: Ensure IAM user has servicediscovery:TagResource permission
# If permission issues occur, tags are set to empty map
resource "aws_service_discovery_private_dns_namespace" "main" {
  name        = "phishing-detection-${var.environment}.local"
  description = "Service discovery namespace for Real-Time-AI-ML-Based-Phishing-Detection-and-Prevention-System"
  vpc         = var.vpc_id
  
  # Use empty tags map to avoid TagResource permission issues
  # Default tags will still be applied via provider default_tags
  tags = {}
  
  lifecycle {
    # Ignore changes to tags_all to prevent permission issues
    ignore_changes = [tags_all]
  }
}

# Outputs
output "cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "cluster_id" {
  description = "ECS cluster ID"
  value       = aws_ecs_cluster.main.id
}

output "alb_dns_name" {
  description = "Application Load Balancer DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_arn" {
  description = "Application Load Balancer ARN"
  value       = aws_lb.main.arn
}

output "service_discovery_namespace_id" {
  description = "Service Discovery namespace ID"
  value       = aws_service_discovery_private_dns_namespace.main.id
}

output "service_discovery_namespace_arn" {
  description = "Service Discovery namespace ARN"
  value       = aws_service_discovery_private_dns_namespace.main.arn
}
