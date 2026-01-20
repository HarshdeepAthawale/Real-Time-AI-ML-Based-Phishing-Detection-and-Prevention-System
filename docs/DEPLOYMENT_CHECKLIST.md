# Deployment Checklist

This checklist covers deployment follow-ups that should be completed when deploying services to production. These items are infrastructure-level configurations that are executed at deployment time, not during development.

## Service Discovery Registration

### Overview
Service discovery registrations in AWS Cloud Map are automatically created when ECS services are deployed with service discovery configuration. This is the expected workflow and doesn't need to be done manually in Terraform.

### Checklist

- [ ] **Verify Service Discovery Namespace**
  - Namespace should be created in Terraform (already done in Phase 1)
  - Location: `infrastructure/terraform/modules/ecs/main.tf`
  - Verify namespace exists: `phishing-detection.local`

- [ ] **Configure ECS Service Discovery**
  - Each ECS service should have service discovery configuration
  - Services to register:
    - `api-gateway`
    - `detection-api`
    - `nlp-service`
    - `url-service`
    - `visual-service`
    - `threat-intel`
    - `extension-api`
    - `sandbox-service`
    - `learning-pipeline`

- [ ] **Verify Service Registration**
  - After ECS deployment, verify services appear in Cloud Map
  - Check service health in Cloud Map console
  - Verify DNS resolution works: `nslookup api-gateway.phishing-detection.local`

### Configuration Example

When deploying ECS services, ensure service discovery is configured:

```hcl
service_registries {
  registry_arn = aws_service_discovery_service.example.arn
  port         = 8000
}
```

## CloudWatch Logging

### Overview
Per-service CloudWatch log groups provide better log isolation and easier troubleshooting. While a general log group exists, per-service groups are recommended for production.

### Checklist

- [ ] **Create Per-Service Log Groups**
  - Log groups should follow pattern: `/phishing-detection/{service-name}-{env}`
  - Services requiring log groups:
    - `/phishing-detection/api-gateway-{env}`
    - `/phishing-detection/detection-api-{env}`
    - `/phishing-detection/nlp-service-{env}`
    - `/phishing-detection/url-service-{env}`
    - `/phishing-detection/visual-service-{env}`
    - `/phishing-detection/threat-intel-{env}`
    - `/phishing-detection/extension-api-{env}`
    - `/phishing-detection/sandbox-service-{env}`
    - `/phishing-detection/learning-pipeline-{env}`

- [ ] **Configure Log Retention**
  - Set retention policy (recommended: 30 days for dev, 90 days for prod)
  - Configure in Terraform or CloudWatch console

- [ ] **Update ECS Task Definitions**
  - Ensure each service logs to its dedicated log group
  - Update `logConfiguration` in task definitions

### Terraform Configuration

Add to `infrastructure/terraform/modules/ecs/main.tf`:

```hcl
resource "aws_cloudwatch_log_group" "service_logs" {
  for_each = toset([
    "api-gateway",
    "detection-api",
    "nlp-service",
    "url-service",
    "visual-service",
    "threat-intel",
    "extension-api",
    "sandbox-service",
    "learning-pipeline"
  ])
  
  name              = "/phishing-detection/${each.key}-${var.environment}"
  retention_in_days = var.environment == "prod" ? 90 : 30
  
  tags = {
    Environment = var.environment
    Service     = each.key
  }
}
```

## CloudWatch Dashboards

### Checklist

- [ ] **Create Service Health Dashboard**
  - Monitor service health endpoints
  - Track response times
  - Monitor error rates

- [ ] **Create ML Service Dashboard**
  - Track inference latency for NLP, URL, and Visual services
  - Monitor model loading status
  - Track cache hit rates

- [ ] **Create Infrastructure Dashboard**
  - ECS service metrics (CPU, memory)
  - RDS connection pool usage
  - Redis cache metrics
  - S3 request metrics

### Dashboard Metrics

Key metrics to include:
- Service health status (up/down)
- Request count and error rate
- Response time (p50, p95, p99)
- ML inference latency
- Database connection pool usage
- Cache hit/miss rates

## CloudWatch Alarms

### Checklist

- [ ] **Service Health Alarms**
  - Alert when service health check fails
  - Alert when service is down for > 5 minutes
  - Configure SNS topic for notifications

- [ ] **Performance Alarms**
  - Alert when response time > threshold (e.g., 5 seconds)
  - Alert when error rate > threshold (e.g., 5%)
  - Alert when ML inference latency > threshold

- [ ] **Infrastructure Alarms**
  - Alert when ECS service CPU > 80%
  - Alert when ECS service memory > 80%
  - Alert when RDS connection pool > 80%
  - Alert when Redis memory > 80%

### Alarm Configuration Example

```hcl
resource "aws_cloudwatch_metric_alarm" "service_health" {
  alarm_name          = "phishing-detection-service-health-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "Alert when service has no healthy hosts"
  
  alarm_actions = [aws_sns_topic.alerts.arn]
}
```

## Deployment Steps

### Pre-Deployment

1. Review Terraform configuration for service discovery
2. Verify CloudWatch log groups exist or will be created
3. Prepare CloudWatch dashboards (optional but recommended)
4. Configure SNS topics for alarms

### During Deployment

1. Deploy ECS services with service discovery enabled
2. Verify services register in Cloud Map
3. Verify logs appear in CloudWatch log groups
4. Test service health endpoints

### Post-Deployment

1. Verify CloudWatch dashboards show metrics
2. Test CloudWatch alarms (trigger test alert)
3. Verify service discovery DNS resolution
4. Monitor logs for any errors

## Notes

- Service discovery registrations happen automatically during ECS deployment
- CloudWatch log groups can be created via Terraform or manually
- Dashboards and alarms are optional but highly recommended for production
- These items are deployment-time configurations, not development blockers

## References

- Phase 1 Infrastructure: `docs/phases/PHASE1_STATUS.md`
- AWS Service Discovery: https://docs.aws.amazon.com/cloud-map/
- CloudWatch Logs: https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/
- CloudWatch Dashboards: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/
