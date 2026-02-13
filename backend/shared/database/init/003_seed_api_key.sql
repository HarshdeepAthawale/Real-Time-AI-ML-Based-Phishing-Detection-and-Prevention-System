-- =========================================
-- Seed Test API Key for Local Dev / Smoke Tests
-- =========================================
-- TEST_API_KEY: testkey_smoke_test_12345
-- Use this key for smoke-test.sh and integration tests.
-- In production, create keys via create-initial-setup.ts or API.
-- =========================================

-- Only seed if no organizations exist (fresh DB)
DO $$
DECLARE
  org_id UUID;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM organizations LIMIT 1) THEN
    -- Create default organization
    INSERT INTO organizations (name, domain, subscription_tier, is_active)
    VALUES (
      'Smoke Test Organization',
      'smoke-test.local',
      'enterprise',
      true
    )
    RETURNING id INTO org_id;

    -- Insert API key: testkey_smoke_test_12345
    -- Hash generated with bcrypt (salt rounds 10)
    INSERT INTO api_keys (organization_id, key_hash, name, permissions, rate_limit)
    VALUES (
      org_id,
      '$2b$10$V0POSI9a8enpAJO1fAsDW.BBkMWoI7FjEnB09xBEbOFaeHHS1zgMa',
      'Smoke Test API Key',
      '["read", "write", "admin"]'::jsonb,
      10000
    );

    RAISE NOTICE 'Seeded org and API key. TEST_API_KEY=testkey_smoke_test_12345';
  END IF;
END $$;
