#!/bin/bash

# Setup script for Threat Intelligence Service test environment
# This script sets up PostgreSQL and Redis for running integration tests

set -e

echo "🔧 Setting up test environment for Threat Intelligence Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if PostgreSQL is running
echo -e "${YELLOW}Checking PostgreSQL...${NC}"
if command -v pg_isready &> /dev/null; then
    if pg_isready -h localhost -p 5432 &> /dev/null; then
        echo -e "${GREEN}✓ PostgreSQL is running${NC}"
    else
        echo -e "${RED}✗ PostgreSQL is not running on localhost:5432${NC}"
        echo "Please start PostgreSQL and try again"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ pg_isready not found, skipping PostgreSQL check${NC}"
fi

# Check if Redis is running
echo -e "${YELLOW}Checking Redis...${NC}"
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✓ Redis is running${NC}"
    else
        echo -e "${RED}✗ Redis is not running${NC}"
        echo "Please start Redis and try again"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ redis-cli not found, skipping Redis check${NC}"
fi

# Get database connection details
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
TEST_DB_NAME="phishing_detection_test"

# Create test database
echo -e "${YELLOW}Creating test database: ${TEST_DB_NAME}...${NC}"
export PGPASSWORD="${DB_PASSWORD}"

# Check if database exists
DB_EXISTS=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -lqt | cut -d \| -f 1 | grep -w "${TEST_DB_NAME}" | wc -l)

if [ "$DB_EXISTS" -eq "0" ]; then
    createdb -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" "${TEST_DB_NAME}" 2>/dev/null || {
        echo -e "${YELLOW}Database might already exist or you don't have permissions. Continuing...${NC}"
    }
    echo -e "${GREEN}✓ Test database created${NC}"
else
    echo -e "${GREEN}✓ Test database already exists${NC}"
fi

# Run migrations
echo -e "${YELLOW}Running database migrations...${NC}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT/backend/shared/database"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Set test database URL
export DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${TEST_DB_NAME}"

# Run migrations
if npm run migration:run 2>/dev/null; then
    echo -e "${GREEN}✓ Migrations completed${NC}"
else
    echo -e "${YELLOW}⚠ Migration command failed, trying alternative method...${NC}"
    # Try running migrations manually
    if [ -f "migrate.ts" ]; then
        npx ts-node migrate.ts || echo -e "${YELLOW}⚠ Could not run migrations automatically${NC}"
    fi
fi

cd - > /dev/null

echo ""
echo -e "${GREEN}✅ Test environment setup complete!${NC}"
echo ""
echo "You can now run tests with:"
echo "  cd backend/core-services/threat-intel"
echo "  npm test"
echo ""
echo "Or run integration tests only:"
echo "  npm run test:integration"
echo ""
