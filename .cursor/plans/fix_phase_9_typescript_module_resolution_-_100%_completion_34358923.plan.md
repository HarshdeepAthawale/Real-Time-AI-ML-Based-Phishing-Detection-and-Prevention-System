---
name: Fix Phase 9 TypeScript Module Resolution - 100% Completion
overview: Fix TypeScript compilation errors preventing 6 test suites from running by properly configuring Jest module resolution for shared database imports. Ensure all 62 tests pass and Phase 9 is 100% complete.
todos:
  - id: "1"
    content: Update jest.config.js moduleNameMapper to resolve ../../shared/* paths
    status: completed
  - id: "2"
    content: Update ts-jest configuration to include paths from tsconfig.json
    status: completed
  - id: "3"
    content: Verify type declarations cover all import paths
    status: completed
  - id: "4"
    content: Test module resolution with single failing test suite
    status: completed
  - id: "5"
    content: Run all integration tests to verify fixes
    status: completed
  - id: "6"
    content: Run complete test suite and verify 100% pass rate
    status: in_progress
  - id: "7"
    content: Generate coverage report and verify 80% threshold met
    status: pending
---

# Fix Phase 9 TypeScript Module Resolution - 100% Completion

## Problem Analysis

Currently, 6 test suites fail due to TypeScript compilation errors when importing shared database modules:

- `extension-auth.middleware.test.ts`
- `rate-limit.middleware.test.ts`  
- All 4 integration route test files (`url-check.routes.test.ts`, `email-scan.routes.test.ts`, `report.routes.test.ts`, `email-client.routes.test.ts`)

**Root Cause**: Jest cannot resolve relative import paths like `../../shared/database` because:

1. Jest's `moduleNameMapper` doesn't map these relative paths
2. ts-jest needs explicit path mapping configuration
3. Type declarations exist but runtime module resolution fails

**Current Status**:

- 54/55 tests passing (98%)
- 5/11 test suites passing completely
- Source code compiles fine (imports work at runtime)
- Only test compilation fails

## Solution Strategy

Fix Jest module resolution by:

1. Updating `jest.config.js` to properly map `../../shared/*` paths
2. Ensuring ts-jest uses the correct tsconfig with paths
3. Verifying all imports resolve correctly
4. Running full test suite to confirm 100% pass rate

## Implementation Steps

### 1. Fix Jest Module Resolution

**File**: `backend/core-services/extension-api/jest.config.js`

Update `moduleNameMapper` to map relative shared module paths:

```javascript
moduleNameMapper: {
  '^@/(.*)$': '<rootDir>/src/$1',
  '^@shared/(.*)$': '<rootDir>/../../shared/$1',
  // Add mapping for relative paths used in source code
  '^(\\.\\./){2}shared/(.*)$': '<rootDir>/../../shared/$2',
},
```

**Alternative approach**: Use regex pattern to match `../../shared/*` imports:

```javascript
moduleNameMapper: {
  '^@/(.*)$': '<rootDir>/src/$1',
  '^(\\.\\./)+shared/(.*)$': '<rootDir>/../../shared/$2',
},
```

### 2. Update ts-jest Configuration

**File**: `backend/core-services/extension-api/jest.config.js`

Ensure ts-jest uses the tsconfig.json that includes paths configuration:

```javascript
transform: {
  '^.+\\.ts$': ['ts-jest', {
    tsconfig: {
      experimentalDecorators: true,
      emitDecoratorMetadata: true,
      strictPropertyInitialization: false,
      skipLibCheck: true,
      // Include paths from tsconfig.json
      baseUrl: '.',
      paths: {
        '../../shared/*': ['../../shared/*']
      }
    },
  }],
},
```

### 3. Verify Type Declarations

**File**: `backend/core-services/extension-api/src/types/shared.d.ts`

Ensure type declarations cover all import paths used in tests. Current declarations cover:

- `../../shared/database` (from src/)
- `../../../shared/database` (from tests/)

No changes needed if paths are correct.

### 4. Test Import Resolution

After configuration changes, verify:

- Source files can import shared modules (already working)
- Test files can import shared modules (needs verification)
- Both relative paths (`../../shared/*`) resolve correctly

### 5. Fix Any Remaining Import Issues

**Files to check**:

- `tests/unit/middleware/extension-auth.middleware.test.ts` - Uses `require()` for shared modules
- `tests/integration/routes/*.test.ts` - Import middleware that imports shared modules

If `require()` approach works but TypeScript complains, ensure:

- Type declarations are correct
- Jest resolves modules at runtime
- TypeScript compilation passes

### 6. Run Full Test Suite

Execute complete test suite:

```bash
cd backend/core-services/extension-api
npm test
```

**Expected Result**:

- All 11 test suites pass
- All 62 tests pass (currently 54 passing + 1 skipped)
- No TypeScript compilation errors
- 100% test pass rate

### 7. Verify Test Coverage

Generate coverage report:

```bash
npm run test:coverage
```

**Expected**: Minimum 80% coverage as per Phase 9 requirements.

## Files to Modify

1. **`backend/core-services/extension-api/jest.config.js`**

   - Update `moduleNameMapper` to handle relative shared module paths
   - Ensure ts-jest configuration includes path mappings

2. **`backend/core-services/extension-api/tsconfig.json`** (if needed)

   - Verify paths configuration is correct
   - Ensure `include` covers test files

## Testing Strategy

1. **Unit Test**: Run a single failing test suite to verify fix
   ```bash
   npm test -- --testPathPattern="extension-auth.middleware"
   ```

2. **Integration Test**: Run integration tests
   ```bash
   npm test -- --testPathPattern="integration"
   ```

3. **Full Suite**: Run all tests
   ```bash
   npm test
   ```

4. **Coverage**: Verify coverage threshold
   ```bash
   npm run test:coverage
   ```


## Success Criteria

- [ ] All 11 test suites compile without TypeScript errors
- [ ] All 62 tests pass (or appropriate number if some are skipped)
- [ ] No module resolution errors in Jest output
- [ ] Test coverage meets 80% threshold
- [ ] Phase 9 marked as 100% complete

## Alternative Solutions (if primary approach fails)

### Option A: Use require() in all test files

- Convert all shared module imports to `require()` in test files
- Keep type declarations for TypeScript
- Less ideal but guaranteed to work

### Option B: Create symlink or workspace

- Set up proper monorepo workspace structure
- Use workspace references for shared modules
- More complex but better long-term solution

### Option C: Copy shared types to test directory

- Create local type definitions in tests
- Avoid importing shared modules in tests
- Not ideal for integration tests that need real modules

## Notes

- Source code imports work fine - this is purely a Jest/TypeScript configuration issue
- Runtime behavior is correct (54 tests passing proves this)
- Focus on making Jest resolve modules the same way TypeScript/Node.js does
- The `moduleNameMapper` regex pattern must match the exact import paths used in code