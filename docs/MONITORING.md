# Monitoring and Alerting

## 1. Service Health Monitoring

### Health Check Endpoints

All services expose a health endpoint:

```
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "uptime": 3600,
  "version": "1.0.0",
  "dependencies": {
    "redis": "connected",
    "mongodb": "connected",
    "nlp_service": "healthy"
  }
}
```

### Health Check Configuration

```yaml
# Docker Compose health checks
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:${PORT}/api/v1/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

### ECS Health Check

```json
{
  "command": ["CMD-SHELL", "curl -f http://localhost:${PORT}/api/v1/health || exit 1"],
  "interval": 30,
  "timeout": 10,
  "retries": 3,
  "startPeriod": 60
}
```

---

## 2. Key Metrics

### Detection API Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `detection.latency.p50` | Median detection latency | < 50ms |
| `detection.latency.p95` | 95th percentile latency | < 100ms |
| `detection.latency.p99` | 99th percentile latency | < 200ms |
| `detection.requests.total` | Total requests per minute | - |
| `detection.threats.detected` | Threats detected per minute | - |
| `detection.errors.rate` | Error rate (5xx responses) | < 1% |

Access latency metrics:
```
GET /api/v1/metrics/latency
```

### ML Service Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `model.inference.latency` | Model inference time | < 30ms |
| `model.confidence.mean` | Average prediction confidence | > 0.8 |
| `model.drift.score` | Model drift score | < 0.15 |
| `model.cache.hit_rate` | Prediction cache hit rate | > 80% |

### Infrastructure Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `cpu.utilization` | CPU usage per service | > 80% for 5 min |
| `memory.utilization` | Memory usage per service | > 85% for 5 min |
| `redis.memory.used` | Redis memory consumption | > 80% of max |
| `redis.connections.active` | Active Redis connections | > 500 |
| `mongodb.connections.current` | Active MongoDB connections | > 100 |
| `mongodb.opcounters.query` | MongoDB queries per second | > 1000 |

---

## 3. CloudWatch Dashboards

### Dashboard: Detection Overview

Widgets:
1. **Request Volume** - Line chart of requests per minute by endpoint
2. **Latency Percentiles** - Line chart of p50/p95/p99 latency
3. **Threat Detection Rate** - Bar chart of threats detected per hour
4. **Error Rate** - Line chart of 4xx and 5xx rates
5. **Confidence Distribution** - Histogram of detection confidence scores
6. **Detection by Type** - Pie chart (email/url/text/sms)

### Dashboard: ML Model Performance

Widgets:
1. **Model Drift Score** - Line chart over time per model
2. **Inference Latency** - Line chart per ML service
3. **Cache Hit Rate** - Line chart for Redis cache
4. **Retraining Events** - Event timeline
5. **Validation Metrics** - TPR, FPR, F1 over time

### Dashboard: Infrastructure

Widgets:
1. **CPU/Memory per Service** - Stacked area chart
2. **ECS Task Count** - Line chart showing scaling events
3. **Redis Memory and Connections** - Dual-axis chart
4. **MongoDB Operations** - Query/insert/update rates
5. **Network I/O** - Bytes in/out per service

---

## 4. Alerting Rules

### Critical Alerts (PagerDuty / immediate response)

```yaml
alerts:
  - name: service_down
    condition: health_check_failures >= 3
    for: 2m
    severity: critical
    action: page_oncall

  - name: high_error_rate
    condition: error_rate_5xx > 5%
    for: 5m
    severity: critical
    action: page_oncall

  - name: detection_latency_critical
    condition: p99_latency > 500ms
    for: 5m
    severity: critical
    action: page_oncall
```

### Warning Alerts (Slack / next business day)

```yaml
alerts:
  - name: high_latency
    condition: p95_latency > 100ms
    for: 10m
    severity: warning
    action: notify_slack

  - name: model_drift_detected
    condition: drift_score > 0.15
    for: 1h
    severity: warning
    action: notify_slack

  - name: high_false_positive_rate
    condition: fpr > 2%
    for: 1h
    severity: warning
    action: notify_slack

  - name: redis_memory_high
    condition: redis_memory_usage > 80%
    for: 15m
    severity: warning
    action: notify_slack

  - name: feed_sync_failed
    condition: feed_sync_error_count > 0
    for: 30m
    severity: warning
    action: notify_slack
```

### Info Alerts (Log only)

```yaml
alerts:
  - name: auto_retrain_triggered
    condition: retrain_event == true
    severity: info
    action: log

  - name: new_ioc_synced
    condition: ioc_sync_count > 100
    severity: info
    action: log
```

---

## 5. Log Aggregation

### Log Format (Structured JSON)

All services output structured JSON logs:

```json
{
  "timestamp": "2024-12-15T10:30:00.000Z",
  "level": "info",
  "service": "detection-api",
  "requestId": "uuid",
  "method": "POST",
  "path": "/api/v1/detect/email",
  "statusCode": 200,
  "latencyMs": 45,
  "apiKey": "key_***masked",
  "isThreat": true,
  "severity": "high",
  "confidence": 0.92
}
```

### Log Retention

| Environment | Retention | Storage |
|-------------|-----------|---------|
| Development | 7 days | Local / stdout |
| Staging | 30 days | CloudWatch Logs |
| Production | 90 days | CloudWatch Logs + S3 archive |
| Government | 365 days | CloudWatch Logs + S3 (encrypted) |

### Log Queries (CloudWatch Insights)

```sql
-- Top 10 detected threat types in last 24h
fields @timestamp, severity, threatType
| filter isThreat = true
| stats count() as cnt by threatType
| sort cnt desc
| limit 10

-- p95 latency per endpoint in last hour
fields @timestamp, path, latencyMs
| stats pct(latencyMs, 95) as p95 by path
| sort p95 desc

-- Error rate by service
fields @timestamp, service, statusCode
| filter statusCode >= 500
| stats count() as errors by service
| sort errors desc

-- False positive reports
fields @timestamp, url, reportedBy
| filter eventType = "false_positive_report"
| sort @timestamp desc
| limit 50
```

---

## 6. Operational Runbook Integration

### Incident Response

1. **Alert fires** -> Check CloudWatch dashboard
2. **Identify affected service** -> Check health endpoint
3. **Review recent changes** -> Check ECS deployment history
4. **Check logs** -> CloudWatch Insights query
5. **Mitigate** -> Scale up / roll back / restart
6. **Post-mortem** -> Document root cause and prevention

### On-Call Checklist

- [ ] All health endpoints returning 200
- [ ] p95 latency < 100ms
- [ ] Error rate < 1%
- [ ] Model drift score < threshold
- [ ] Threat feed sync successful in last cycle
- [ ] Redis memory usage < 80%
- [ ] No pending security alerts
