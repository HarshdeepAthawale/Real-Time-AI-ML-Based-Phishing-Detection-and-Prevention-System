# Security Review Checklist

## 1. Authentication and Authorization

### API Key Management
- [x] All public endpoints require `X-API-Key` header
- [x] API keys validated via middleware before route handlers
- [x] API keys hashed (not stored in plaintext) in database
- [ ] API key rotation mechanism (recommended: 90-day rotation)
- [x] Rate limiting per API key to prevent abuse

### Extension Authentication
- [x] Browser extension sends `X-Extension-Id` header
- [x] Extension ID validated against registered extensions
- [x] Extension API has separate rate limits from main API

### OAuth (Gmail/Outlook)
- [ ] OAuth tokens stored encrypted at rest
- [ ] Refresh tokens rotated on use
- [ ] Minimal OAuth scopes requested (read-only email access)
- [x] OAuth state parameter used to prevent CSRF

---

## 2. Input Validation

### Detection Endpoints
- [x] Zod schema validation on all request bodies
- [x] URL format validation before analysis
- [x] Email content length limits enforced
- [x] SMS message length limits enforced
- [x] `organizationId` validated as UUID format

### ML Service Inputs
- [x] Text input truncated to max token length (256 tokens)
- [x] URL length capped before parsing
- [x] Image size validated before CNN processing
- [x] Request body size limited via Express middleware

---

## 3. Secrets Management

### Current State
- [x] No API keys or secrets committed to source code
- [x] Environment variables used for all secrets
- [x] `.env` files in `.gitignore`

### Production Recommendations
- [ ] Use AWS Secrets Manager or SSM Parameter Store
- [ ] Enable secret rotation for database credentials
- [ ] Encrypt secrets at rest with KMS
- [ ] Audit secret access via CloudTrail

### Secrets Inventory

| Secret | Storage | Rotation |
|--------|---------|----------|
| `API_KEY_SECRET` | Env var / SSM | Manual |
| `JWT_SECRET` | Env var / SSM | Manual |
| `MONGODB_URI` | Env var / SSM | Manual |
| `MISP_API_KEY` | Env var / SSM | Per vendor policy |
| `OTX_API_KEY` | Env var / SSM | Per vendor policy |
| `GMAIL_CLIENT_SECRET` | Env var / SSM | Manual |
| `OUTLOOK_CLIENT_SECRET` | Env var / SSM | Manual |
| `CUCKOO_API_URL` | Env var / SSM | N/A |
| `ANYRUN_API_KEY` | Env var / SSM | Per vendor policy |

---

## 4. Network Security

### CORS
- [x] CORS restricted to configured origins (`CORS_ORIGINS` env var)
- [x] Credentials mode disabled for public endpoints
- [x] Preflight caching enabled

### Service-to-Service Communication
- [x] Internal ML services not exposed publicly (internal network only)
- [ ] mTLS between services (recommended for government profile)
- [x] Service discovery via environment variable URLs

### Rate Limiting
- [x] Per-API-key rate limits on detection endpoints (100 req/min default)
- [x] Per-API-key rate limits on extension endpoints (200 req/min default)
- [x] Threat intel endpoints rate limited (50 req/min default)
- [x] Redis-backed rate limiter for distributed deployments

---

## 5. Data Protection

### Data at Rest
- [x] MongoDB data encrypted at rest (Atlas/DocumentDB default)
- [ ] S3 bucket encryption enabled (SSE-S3 or SSE-KMS)
- [ ] Redis data not persisted to disk (cache-only mode)

### Data in Transit
- [x] HTTPS enforced on all public endpoints
- [x] TLS 1.2+ required
- [ ] Certificate pinning in browser extension (optional)

### PII Handling
- [x] Email content not stored after analysis (unless feedback submitted)
- [x] Detection results store threat indicators, not full content
- [ ] Log redaction for PII (email addresses, phone numbers)
- [ ] Data retention policy enforced (auto-delete after N days)

---

## 6. Dependency Security

### Node.js
- [x] `package-lock.json` committed for deterministic builds
- [ ] `npm audit` run in CI pipeline
- [ ] Dependabot or Snyk configured for vulnerability alerts

### Python
- [x] Pinned dependency versions in `requirements.txt`
- [ ] `pip-audit` or `safety` run in CI pipeline
- [ ] No known critical vulnerabilities in dependencies

### Docker
- [x] Base images use specific tags (not `latest`)
- [ ] Images scanned with Trivy or similar
- [ ] Non-root user in Dockerfiles

---

## 7. Logging and Monitoring

### Audit Logging
- [x] Detection requests logged with timestamp, source, result
- [x] API key usage logged per request
- [ ] Admin actions logged (key creation, config changes)
- [ ] Log integrity (append-only, tamper-evident)

### Security Monitoring
- [ ] Alert on > 10 failed auth attempts per minute
- [ ] Alert on unusual traffic patterns (volume spikes)
- [ ] Alert on model drift (accuracy degradation)
- [x] Latency monitoring with p50/p95/p99 metrics

---

## 8. Injection and OWASP Top 10

| Risk | Status | Notes |
|------|--------|-------|
| SQL/NoSQL Injection | Mitigated | Mongoose ODM with schema validation |
| XSS | Mitigated | API-only (no HTML rendering); extension uses `textContent` |
| CSRF | Mitigated | API key auth (no cookies); OAuth state param |
| SSRF | Partial | URL service follows redirects; max hop limit enforced |
| Broken Auth | Mitigated | API key + rate limiting on all endpoints |
| Security Misconfiguration | Review needed | Default configs should be hardened for production |
| Insecure Deserialization | Mitigated | JSON-only APIs; no object deserialization |
| Insufficient Logging | Partial | Request logging exists; audit trail needs enhancement |
| Command Injection | Mitigated | No shell commands executed from user input |
| Path Traversal | Mitigated | No file path construction from user input |

---

## 9. Browser Extension Security

- [x] Manifest V3 (service worker, no persistent background)
- [x] Minimal permissions requested
- [x] Content Security Policy defined in manifest
- [x] No `eval()` or dynamic code execution
- [x] External API calls only to configured backend URL
- [x] Blocked page served from extension bundle (not remote)
- [x] User can allowlist URLs to bypass blocking

---

## 10. Action Items (Priority Order)

1. **High:** Enable `npm audit` and `pip-audit` in CI
2. **High:** Configure AWS Secrets Manager for production secrets
3. **High:** Add PII redaction to log pipeline
4. **Medium:** Enable mTLS for government deployments
5. **Medium:** Configure Dependabot for dependency updates
6. **Medium:** Add audit logging for admin actions
7. **Low:** Implement API key rotation mechanism
8. **Low:** Add certificate pinning to browser extension
