#!/usr/bin/env bash
# Start the full stack locally using Docker Compose (recommended).
# Usage: ./scripts/start-local.sh
# Requires: Docker, and .env at project root with POSTGRES_PASSWORD set.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-change_me_secure_password}"

echo "Starting full stack (frontend + backend + ML services) with Docker Compose..."
echo ""
docker compose up --build

# When you Ctrl+C, compose will stop. To run in background instead, use:
#   docker compose up --build -d
# Then: docker compose down
# Access: Frontend http://localhost:3080  |  API Gateway http://localhost:3000
