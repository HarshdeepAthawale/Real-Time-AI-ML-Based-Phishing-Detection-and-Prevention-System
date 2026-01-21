import { Request, Response, NextFunction } from 'express';
import { extensionAuthMiddleware, ExtensionRequest } from '../../../src/middleware/extension-auth.middleware';
import { connectTestDatabase, getTestDatabase, truncateTables } from '../../helpers/test-db';
import { createTestApiKey, createExpiredApiKey, createRevokedApiKey, createStandardTestFixtures, TEST_API_KEY } from '../../fixtures/api-keys';
// Use require to avoid TypeScript path issues
const sharedDb = require('../../../shared/database');
const getPostgreSQL = sharedDb.getPostgreSQL;
const { ApiKey } = require('../../../shared/database/models');

describe('extensionAuthMiddleware', () => {
  let mockRequest: Partial<ExtensionRequest>;
  let mockResponse: Partial<Response>;
  let mockNext: NextFunction;
  let dataSource: any;
  let standardFixtures: any;

  beforeAll(async () => {
    // Ensure DATABASE_URL is set for shared connection
    process.env.DATABASE_URL = process.env.TEST_DATABASE_URL || 'postgresql://postgres:postgres@localhost:5433/phishing_detection_test';
    dataSource = await connectTestDatabase();
  });

  beforeEach(async () => {
    // Clean up database before each test
    await truncateTables();
    
    // Reset mocks
    jest.clearAllMocks();
    
    // Create test fixtures
    standardFixtures = await createStandardTestFixtures(dataSource);

    mockRequest = {
      headers: {},
      path: '/test',
      ip: '127.0.0.1',
      app: {} as any,
    };

    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };

    mockNext = jest.fn();
  });

  describe('API key validation', () => {
    it('should reject request without API key', async () => {
      mockRequest.headers = {};

      await extensionAuthMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            message: 'API key required',
          }),
        })
      );
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should reject invalid API key format', async () => {
      mockRequest.headers = {
        'x-api-key': 'invalid',
      };

      await extensionAuthMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            message: 'Invalid API key format',
          }),
        })
      );
    });

    it('should authenticate valid API key', async () => {
      mockRequest.headers = {
        'x-api-key': standardFixtures.fullApiKey,
        'x-extension-id': 'ext-123',
      };

      await extensionAuthMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockNext).toHaveBeenCalled();
      expect(mockRequest.apiKey).toBe(standardFixtures.fullApiKey);
      expect(mockRequest.extensionId).toBe('ext-123');
    });

    it('should reject revoked API key', async () => {
      const { fullKey } = await createRevokedApiKey(
        dataSource,
        standardFixtures.organization.id,
        'revokedkey_test1234567890'
      );

      mockRequest.headers = {
        'x-api-key': fullKey,
      };

      await extensionAuthMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            message: 'API key has been revoked',
          }),
        })
      );
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should reject expired API key', async () => {
      const { fullKey } = await createExpiredApiKey(
        dataSource,
        standardFixtures.organization.id,
        'expiredkey_test1234567890'
      );

      mockRequest.headers = {
        'x-api-key': fullKey,
      };

      await extensionAuthMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          error: expect.objectContaining({
            message: 'API key has expired',
          }),
        })
      );
    });

    it('should reject API key with invalid hash', async () => {
      // Create API key with one hash, but try to use different key
      const { fullKey } = await createTestApiKey(
        dataSource,
        standardFixtures.organization.id,
        'validkey_test1234567890'
      );

      // Try to use wrong key
      mockRequest.headers = {
        'x-api-key': 'validkey_wrongkey1234567890',
      };

      await extensionAuthMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should reject non-existent API key', async () => {
      const apiKey = 'nonexistent_test1234567890';

      mockRequest.headers = {
        'x-api-key': apiKey,
      };

      await extensionAuthMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should update last_used_at timestamp', async () => {
      const { apiKey, fullKey } = await createTestApiKey(
        dataSource,
        standardFixtures.organization.id,
        'updatekey_test1234567890'
      );

      mockRequest.headers = {
        'x-api-key': fullKey,
      };

      const initialLastUsed = apiKey.last_used_at;

      await extensionAuthMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockNext).toHaveBeenCalled();
      
      // Verify last_used_at was updated
      const db = getPostgreSQL();
      const apiKeyRepository = db.getRepository(ApiKey);
      const updated = await apiKeyRepository.findOne({ where: { id: apiKey.id } });
      expect(updated?.last_used_at).not.toBeNull();
      if (initialLastUsed && updated?.last_used_at) {
        expect(updated.last_used_at.getTime()).toBeGreaterThanOrEqual(initialLastUsed.getTime());
      }
    });

    it('should use organization ID from API key if not provided in header', async () => {
      mockRequest.headers = {
        'x-api-key': standardFixtures.fullApiKey,
      };

      await extensionAuthMiddleware(
        mockRequest as ExtensionRequest,
        mockResponse as Response,
        mockNext
      );

      expect(mockRequest.organizationId).toBe(standardFixtures.organization.id);
    });
  });
});
