#!/bin/bash
# Pre-push checks: security scan + LFS verification
# Run before: git push
# Usage: ./scripts/pre-push-checks.sh
#   --no-prompt  Fail on npm audit without asking (for CI/hooks)

set -e
NO_PROMPT=false
[[ "${1:-}" == "--no-prompt" ]] && NO_PROMPT=true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILED=0

echo "=== Pre-push checks: Security + LFS ==="
echo ""

# --- 1. Security: npm audit ---
echo "1. Security: npm audit (root)..."
set +e
npm audit --audit-level=high 2>/dev/null
AUDIT_EXIT=$?
set -e
if [ $AUDIT_EXIT -eq 0 ]; then
  echo -e "${GREEN}✓ npm audit passed${NC}"
else
  echo -e "${YELLOW}⚠ npm audit found vulnerabilities. Run 'npm audit fix' to auto-fix.${NC}"
  if $NO_PROMPT; then
    FAILED=1
  else
    echo "  Continue anyway? (y/N)"
    read -r ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
      FAILED=1
    fi
  fi
fi
echo ""

# --- 2. Security: secret patterns in staged files ---
echo "2. Security: scanning staged files for secrets..."
STAGED=$(git diff --cached --name-only 2>/dev/null | grep -v '^$' || true)
if [ -n "$STAGED" ]; then
  # Simple patterns: AWS keys, OpenAI keys, private keys
  SECRETS=$(echo "$STAGED" | xargs grep -l -E '(AKIA[0-9A-Z]{16}|sk-[a-zA-Z0-9]{32,}|-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----)' 2>/dev/null || true)
  if [ -n "$SECRETS" ]; then
    echo -e "${RED}✗ Possible secrets detected in: $SECRETS${NC}"
    FAILED=1
  else
    echo -e "${GREEN}✓ No obvious secret patterns in staged files${NC}"
  fi
else
  echo -e "${GREEN}✓ No staged files to scan${NC}"
fi
echo ""

# --- 3. LFS check ---
echo "3. LFS check..."
if command -v git-lfs &>/dev/null; then
  if git lfs env 2>/dev/null | grep -q "git config filter.lfs"; then
    echo "  LFS configured. Verifying..."
    git lfs status 2>/dev/null || true
    echo -e "${GREEN}✓ LFS OK${NC}"
  else
    echo "  LFS not configured for this repo (optional for this project)"
  fi
else
  echo "  git-lfs not installed (optional)"
fi

# Check for large files in staged changes (>1MB)
LARGE_STAGED=""
for path in $(git diff --cached --name-only 2>/dev/null); do
  [ -f "$path" ] || continue
  size=$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path" 2>/dev/null || echo 0)
  [ "$size" -gt 1048576 ] 2>/dev/null && LARGE_STAGED="${LARGE_STAGED}${path} ($((size/1048576))MB)\n"
done
if [ -n "$LARGE_STAGED" ]; then
  echo -e "${YELLOW}⚠ Large files staged (>1MB):${NC}"
  echo -e "$LARGE_STAGED"
  echo "  Consider using Git LFS: git lfs install && git lfs track '*.bin' etc."
else
  echo -e "${GREEN}✓ No large files (>1MB) staged${NC}"
fi
echo ""

# --- Summary ---
if [ $FAILED -eq 1 ]; then
  echo -e "${RED}Pre-push checks FAILED. Fix issues before pushing.${NC}"
  exit 1
fi
echo -e "${GREEN}Pre-push checks passed. Safe to push.${NC}"
