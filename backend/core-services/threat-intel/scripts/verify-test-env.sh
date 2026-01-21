#!/bin/bash

# Verify test environment is set up correctly

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Verifying test environment..."

# Check PostgreSQL
echo -n "Checking PostgreSQL... "
if command -v pg_isready &> /dev/null && pg_isready -h localhost -p 5432 &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  PostgreSQL is not running or not accessible"
    exit 1
fi

# Check Redis
echo -n "Checking Redis... "
if command -v redis-cli &> /dev/null && redis-cli ping &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  Redis is not running or not accessible"
    exit 1
fi

# Check test database
echo -n "Checking test database... "
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
TEST_DB_NAME="phishing_detection_test"
export PGPASSWORD="${DB_PASSWORD}"

DB_EXISTS=$(psql -h localhost -U "${DB_USER}" -lqt 2>/dev/null | cut -d \| -f 1 | grep -w "${TEST_DB_NAME}" | wc -l)

if [ "$DB_EXISTS" -eq "1" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  Test database '${TEST_DB_NAME}' does not exist"
    echo "  Run: npm run test:setup"
    exit 1
fi

# Check if tables exist
echo -n "Checking database tables... "
TABLE_COUNT=$(psql -h localhost -U "${DB_USER}" -d "${TEST_DB_NAME}" -tAc "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null)

if [ "$TABLE_COUNT" -gt "0" ]; then
    echo -e "${GREEN}✓ (${TABLE_COUNT} tables)${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  No tables found in test database"
    echo "  Run: npm run test:setup"
    exit 1
fi

# Check required tables
echo -n "Checking required tables... "
REQUIRED_TABLES=("iocs" "threat_intelligence_feeds" "ioc_matches")
MISSING_TABLES=()

for table in "${REQUIRED_TABLES[@]}"; do
    EXISTS=$(psql -h localhost -U "${DB_USER}" -d "${TEST_DB_NAME}" -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '${table}');" 2>/dev/null)
    if [ "$EXISTS" != "t" ]; then
        MISSING_TABLES+=("${table}")
    fi
done

if [ ${#MISSING_TABLES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  Missing tables: ${MISSING_TABLES[*]}"
    echo "  Run: npm run test:setup"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Test environment is ready!${NC}"
echo ""
echo "You can now run:"
echo "  npm test              # Run all tests"
echo "  npm run test:unit     # Run unit tests only"
echo "  npm run test:integration  # Run integration tests only"
