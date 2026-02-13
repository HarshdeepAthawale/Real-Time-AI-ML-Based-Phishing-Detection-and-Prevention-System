# ECS Task Definitions and Services
# Requires: ECR images pushed by CI (phishing-detection-$service:latest)

variable "db_host" {
  description = "RDS host"
  type        = string
  default     = ""
}

variable "db_port" {
  description = "RDS port"
  type        = string
  default     = "5432"
}

variable "db_name" {
  description = "RDS database name"
  type        = string
  default     = "phishing_detection"
}

variable "db_username" {
  description = "RDS username"
  type        = string
  default     = "postgres"
}

variable "db_password" {
  description = "RDS password"
  type        = string
  sensitive   = true
  default     = ""
}

variable "redis_endpoint" {
  description = "Redis endpoint (host:port)"
  type        = string
  default     = ""
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  ecr_registry = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${data.aws_region.current.name}.amazonaws.com"
  db_url       = "postgresql://${var.db_username}:${var.db_password}@${var.db_host}:${var.db_port}/${var.db_name}"
  ns_name      = "phishing-detection-${var.environment}.local"
}

# Service Discovery - detection-api
resource "aws_service_discovery_service" "detection_api" {
  name = "detection-api"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# Service Discovery - threat-intel
resource "aws_service_discovery_service" "threat_intel" {
  name = "threat-intel"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# Service Discovery - extension-api
resource "aws_service_discovery_service" "extension_api" {
  name = "extension-api"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# Service Discovery - sandbox-service
resource "aws_service_discovery_service" "sandbox_service" {
  name = "sandbox-service"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# Service Discovery - nlp-service
resource "aws_service_discovery_service" "nlp_service" {
  name = "nlp-service"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# Service Discovery - url-service
resource "aws_service_discovery_service" "url_service" {
  name = "url-service"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# Service Discovery - visual-service
resource "aws_service_discovery_service" "visual_service" {
  name = "visual-service"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      ttl  = 10
      type = "A"
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# API Gateway Task Definition
resource "aws_ecs_task_definition" "api_gateway" {
  family                   = "phishing-detection-api-gateway-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 256
  memory                   = 512

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "api-gateway"
      image     = "${local.ecr_registry}/phishing-detection-api-gateway:latest"
      essential = true

      portMappings = [
        { containerPort = 3000, protocol = "tcp" }
      ]

      environment = [
        { name = "NODE_ENV", value = var.environment },
        { name = "PORT", value = "3000" },
        { name = "DATABASE_URL", value = local.db_url },
        { name = "REDIS_URL", value = "redis://${var.redis_endpoint}" },
        { name = "DETECTION_API_URL", value = "http://detection-api.${local.ns_name}:3001" },
        { name = "THREAT_INTEL_URL", value = "http://threat-intel.${local.ns_name}:3002" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api_gateway.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:3000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# Detection API Task Definition
resource "aws_ecs_task_definition" "detection_api" {
  family                   = "phishing-detection-detection-api-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "detection-api"
      image     = "${local.ecr_registry}/phishing-detection-detection-api:latest"
      essential = true

      portMappings = [
        { containerPort = 3001, protocol = "tcp" }
      ]

      environment = [
        { name = "NODE_ENV", value = var.environment },
        { name = "PORT", value = "3001" },
        { name = "DATABASE_URL", value = local.db_url },
        { name = "REDIS_URL", value = "redis://${var.redis_endpoint}" },
        { name = "THREAT_INTEL_URL", value = "http://threat-intel.${local.ns_name}:3002" },
        { name = "NLP_SERVICE_URL", value = "http://nlp-service.${local.ns_name}:8000" },
        { name = "URL_SERVICE_URL", value = "http://url-service.${local.ns_name}:8001" },
        { name = "VISUAL_SERVICE_URL", value = "http://visual-service.${local.ns_name}:8002" },
        { name = "SANDBOX_SERVICE_URL", value = "http://sandbox-service.${local.ns_name}:3004" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.detection_api.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:3001/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# Threat Intel Task Definition
resource "aws_ecs_task_definition" "threat_intel" {
  family                   = "phishing-detection-threat-intel-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 256
  memory                   = 512

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "threat-intel"
      image     = "${local.ecr_registry}/phishing-detection-threat-intel:latest"
      essential = true

      portMappings = [
        { containerPort = 3002, protocol = "tcp" }
      ]

      environment = [
        { name = "NODE_ENV", value = var.environment },
        { name = "PORT", value = "3002" },
        { name = "DATABASE_URL", value = local.db_url },
        { name = "REDIS_URL", value = "redis://${var.redis_endpoint}" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.threat_intel.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:3002/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# Extension API Task Definition
resource "aws_ecs_task_definition" "extension_api" {
  family                   = "phishing-detection-extension-api-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 256
  memory                   = 512

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "extension-api"
      image     = "${local.ecr_registry}/phishing-detection-extension-api:latest"
      essential = true

      portMappings = [
        { containerPort = 3003, protocol = "tcp" }
      ]

      environment = [
        { name = "NODE_ENV", value = var.environment },
        { name = "PORT", value = "3003" },
        { name = "DATABASE_URL", value = local.db_url },
        { name = "REDIS_URL", value = "redis://${var.redis_endpoint}" },
        { name = "DETECTION_API_URL", value = "http://detection-api.${local.ns_name}:3001" },
        { name = "THREAT_INTEL_URL", value = "http://threat-intel.${local.ns_name}:3002" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "extension-api"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:3003/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# Sandbox Service Task Definition
resource "aws_ecs_task_definition" "sandbox_service" {
  family                   = "phishing-detection-sandbox-service-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "sandbox-service"
      image     = "${local.ecr_registry}/phishing-detection-sandbox-service:latest"
      essential = true

      portMappings = [
        { containerPort = 3004, protocol = "tcp" }
      ]

      environment = [
        { name = "NODE_ENV", value = var.environment },
        { name = "PORT", value = "3004" },
        { name = "DATABASE_URL", value = local.db_url },
        { name = "REDIS_URL", value = "redis://${var.redis_endpoint}" },
        { name = "DETECTION_API_URL", value = "http://detection-api.${local.ns_name}:3001" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "sandbox-service"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:3004/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# NLP Service Task Definition (Python)
resource "aws_ecs_task_definition" "nlp_service" {
  family                   = "phishing-detection-nlp-service-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "nlp-service"
      image     = "${local.ecr_registry}/phishing-detection-nlp-service:latest"
      essential = true

      portMappings = [
        { containerPort = 8000, protocol = "tcp" }
      ]

      environment = [
        { name = "PORT", value = "8000" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ml_services.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "nlp-service"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 90
      }
    }
  ])
}

# URL Service Task Definition (Python)
resource "aws_ecs_task_definition" "url_service" {
  family                   = "phishing-detection-url-service-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "url-service"
      image     = "${local.ecr_registry}/phishing-detection-url-service:latest"
      essential = true

      portMappings = [
        { containerPort = 8001, protocol = "tcp" }
      ]

      environment = [
        { name = "PORT", value = "8001" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ml_services.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "url-service"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8001/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 90
      }
    }
  ])
}

# Visual Service Task Definition (Python)
resource "aws_ecs_task_definition" "visual_service" {
  family                   = "phishing-detection-visual-service-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 512
  memory                   = 1024

  execution_role_arn = aws_iam_role.ecs_task_execution.arn
  task_role_arn      = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "visual-service"
      image     = "${local.ecr_registry}/phishing-detection-visual-service:latest"
      essential = true

      portMappings = [
        { containerPort = 8002, protocol = "tcp" }
      ]

      environment = [
        { name = "PORT", value = "8002" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ml_services.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "visual-service"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8002/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 90
      }
    }
  ])
}

# ECS Services
resource "aws_ecs_service" "api_gateway" {
  name            = "phishing-detection-api-gateway-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api_gateway.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.security_group_id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api_gateway.arn
    container_name  = "api-gateway"
    container_port  = 3000
  }

  depends_on = [
    aws_lb_listener.main
  ]
}

resource "aws_ecs_service" "detection_api" {
  name            = "phishing-detection-detection-api-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.detection_api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.security_group_id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.detection_api.arn
  }
}

resource "aws_ecs_service" "threat_intel" {
  name            = "phishing-detection-threat-intel-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.threat_intel.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.security_group_id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.threat_intel.arn
  }
}

resource "aws_ecs_service" "extension_api" {
  name            = "phishing-detection-extension-api-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.extension_api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.security_group_id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.extension_api.arn
  }
}

resource "aws_ecs_service" "sandbox_service" {
  name            = "phishing-detection-sandbox-service-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.sandbox_service.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.security_group_id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.sandbox_service.arn
  }
}

resource "aws_ecs_service" "nlp_service" {
  name            = "phishing-detection-nlp-service-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.nlp_service.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.security_group_id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.nlp_service.arn
  }
}

resource "aws_ecs_service" "url_service" {
  name            = "phishing-detection-url-service-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.url_service.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.security_group_id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.url_service.arn
  }
}

resource "aws_ecs_service" "visual_service" {
  name            = "phishing-detection-visual-service-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.visual_service.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [var.security_group_id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.visual_service.arn
  }
}
