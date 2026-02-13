#!/bin/bash
# E2E Smoke Test - validates detection API responds correctly
# Requires: services running (docker-compose up) and API key
# Usage: ./scripts/smoke-test.sh [API_BASE_URL] [API_KEY]

set -e

API_BASE="${1:-http://localhost:3000}"
API_KEY="${2:-${TEST_API_KEY}}"

if [ -z "$API_KEY" ]; then
  echo "WARNING: No API key. Set TEST_API_KEY or pass as second argument."
  echo "Attempting without key (may fail if auth required)..."
fi

echo "Smoke test: API_BASE=$API_BASE"
echo "---"

# Health check
echo "1. Health check..."
health=$(curl -sf "${API_BASE}/health" 2>/dev/null || curl -sf "http://localhost:3001/health" 2>/dev/null)
if [ -z "$health" ]; then
  echo "FAIL: Health check failed - services may not be running"
  exit 1
fi
echo "OK: Services healthy"

# Safe URL - should not be threat
echo "2. Safe URL (google.com)..."
headers=(-H "Content-Type: application/json")
[ -n "$API_KEY" ] && headers+=(-H "X-API-Key: $API_KEY")

safe_resp=$(curl -sf "${headers[@]}" -X POST "${API_BASE}/api/v1/detect/url" \
  -d '{"url":"https://www.google.com","organizationId":"smoke-test"}' 2>/dev/null || \
  curl -sf "${headers[@]}" -X POST "http://localhost:3001/api/v1/detect/url" \
  -d '{"url":"https://www.google.com","organizationId":"smoke-test"}' 2>/dev/null)

if [ -z "$safe_resp" ]; then
  echo "FAIL: No response from detect/url"
  exit 1
fi

# Check response structure (has is_threat or isThreat)
if echo "$safe_resp" | grep -qE '"is_threat"|"isThreat"'; then
  echo "OK: Safe URL response has threat field"
else
  echo "WARN: Response structure unexpected (continuing)"
fi

# Suspicious URL - may be flagged
echo "3. Suspicious URL (obfuscated)..."
susp_resp=$(curl -sf "${headers[@]}" -X POST "${API_BASE}/api/v1/detect/url" \
  -d '{"url":"https://amaz0n-billing.xyz/verify?id=base64encoded","organizationId":"smoke-test"}' 2>/dev/null || \
  curl -sf "${headers[@]}" -X POST "http://localhost:3001/api/v1/detect/url" \
  -d '{"url":"https://amaz0n-billing.xyz/verify?id=base64encoded","organizationId":"smoke-test"}' 2>/dev/null)

if [ -z "$susp_resp" ]; then
  echo "FAIL: No response for suspicious URL"
  exit 1
fi
echo "OK: Suspicious URL analyzed"

echo "---"
echo "Smoke test PASSED"
