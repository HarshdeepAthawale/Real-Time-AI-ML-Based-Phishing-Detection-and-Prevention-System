import bcrypt from 'bcrypt';
import { DataSource } from 'typeorm';
import { Organization } from '../../../../shared/database/models/Organization';
import { User } from '../../../../shared/database/models/User';
import { ApiKey } from '../../../../shared/database/models/ApiKey';

/**
 * Create a test organization
 */
export const createTestOrganization = async (
  dataSource: DataSource,
  overrides?: Partial<Organization>
): Promise<Organization> => {
  const orgRepository = dataSource.getRepository(Organization);
  
  // Generate unique domain to avoid conflicts
  const uniqueDomain = overrides?.domain || `test-${Date.now()}-${Math.random().toString(36).substring(7)}.example.com`;
  
  const organization = orgRepository.create({
    name: `Test Organization ${Date.now()}`,
    domain: uniqueDomain,
    plan: 'free',
    max_users: 10,
    max_api_calls_per_day: 10000,
    ...overrides,
  });
  
  return await orgRepository.save(organization);
};

/**
 * Create a test user
 */
export const createTestUser = async (
  dataSource: DataSource,
  organizationId: string,
  overrides?: Partial<User>
): Promise<User> => {
  const userRepository = dataSource.getRepository(User);
  
  const passwordHash = await bcrypt.hash('testpassword123', 10);
  
  // Generate unique email to avoid conflicts
  const uniqueEmail = overrides?.email || `test-${Date.now()}-${Math.random().toString(36).substring(7)}@example.com`;
  
  const user = userRepository.create({
    organization_id: organizationId,
    email: uniqueEmail,
    password_hash: passwordHash,
    first_name: 'Test',
    last_name: 'User',
    role: 'user',
    is_active: true,
    ...overrides,
  });
  
  return await userRepository.save(user);
};

/**
 * Create a test API key
 */
export const createTestApiKey = async (
  dataSource: DataSource,
  organizationId: string,
  apiKeyString: string,
  overrides?: Partial<ApiKey>
): Promise<{ apiKey: ApiKey; fullKey: string }> => {
  const apiKeyRepository = dataSource.getRepository(ApiKey);
  
  // Extract prefix
  const keyPrefix = apiKeyString.includes('_') 
    ? apiKeyString.split('_')[0] 
    : apiKeyString.substring(0, 8);
  
  // Hash the full API key
  const keyHash = await bcrypt.hash(apiKeyString, 10);
  
  const apiKey = apiKeyRepository.create({
    organization_id: organizationId,
    key_prefix: keyPrefix,
    key_hash: keyHash,
    name: 'Test API Key',
    permissions: {},
    rate_limit_per_minute: 100,
    expires_at: null,
    revoked_at: null,
    last_used_at: null,
    ...overrides,
  });
  
  const saved = await apiKeyRepository.save(apiKey);
  
  return {
    apiKey: saved,
    fullKey: apiKeyString,
  };
};

/**
 * Create a test API key that is expired
 */
export const createExpiredApiKey = async (
  dataSource: DataSource,
  organizationId: string,
  apiKeyString: string
): Promise<{ apiKey: ApiKey; fullKey: string }> => {
  const expiredDate = new Date();
  expiredDate.setDate(expiredDate.getDate() - 1); // Yesterday
  
  return createTestApiKey(dataSource, organizationId, apiKeyString, {
    expires_at: expiredDate,
  });
};

/**
 * Create a test API key that is revoked
 */
export const createRevokedApiKey = async (
  dataSource: DataSource,
  organizationId: string,
  apiKeyString: string
): Promise<{ apiKey: ApiKey; fullKey: string }> => {
  return createTestApiKey(dataSource, organizationId, apiKeyString, {
    revoked_at: new Date(),
  });
};

/**
 * Standard test API key for use in tests
 */
export const TEST_API_KEY = 'testkey_abcdefghijklmnopqrstuvwxyz1234567890';

/**
 * Create standard test fixtures (organization, user, API key)
 */
export const createStandardTestFixtures = async (
  dataSource: DataSource
): Promise<{
  organization: Organization;
  user: User;
  apiKey: ApiKey;
  fullApiKey: string;
}> => {
  // Create organization first and ensure it's saved
  const organization = await createTestOrganization(dataSource);
  
  // Verify organization exists before creating dependent entities
  const orgRepository = dataSource.getRepository(Organization);
  const savedOrg = await orgRepository.findOne({ where: { id: organization.id } });
  if (!savedOrg) {
    throw new Error('Organization not found after creation');
  }
  
  const user = await createTestUser(dataSource, savedOrg.id);
  const { apiKey, fullKey } = await createTestApiKey(
    dataSource,
    savedOrg.id, // Use savedOrg.id to ensure it exists
    TEST_API_KEY
  );
  
  return {
    organization: savedOrg,
    user,
    apiKey,
    fullApiKey: fullKey,
  };
};
