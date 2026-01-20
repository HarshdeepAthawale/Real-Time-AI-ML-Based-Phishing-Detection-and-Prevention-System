import { Request, Response, NextFunction } from 'express';
import { authMiddleware, AuthenticatedRequest } from '../../../src/middleware/auth.middleware';

jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('authMiddleware', () => {
  let mockRequest: Partial<AuthenticatedRequest>;
  let mockResponse: Partial<Response>;
  let nextFunction: NextFunction;

  beforeEach(() => {
    mockRequest = {
      headers: {},
      path: '/api/v1/detect/email',
    };

    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };

    nextFunction = jest.fn();
  });

  it('should call next() when API key is provided', () => {
    mockRequest.headers = {
      'x-api-key': 'test-api-key',
    };

    authMiddleware(
      mockRequest as AuthenticatedRequest,
      mockResponse as Response,
      nextFunction
    );

    expect(nextFunction).toHaveBeenCalled();
    expect(mockRequest.apiKey).toBe('test-api-key');
    expect(mockRequest.apiKeyId).toBe('temp-id');
    expect(mockResponse.status).not.toHaveBeenCalled();
  });

  it('should return 401 when API key is missing', () => {
    mockRequest.headers = {};

    authMiddleware(
      mockRequest as AuthenticatedRequest,
      mockResponse as Response,
      nextFunction
    );

    expect(nextFunction).not.toHaveBeenCalled();
    expect(mockResponse.status).toHaveBeenCalledWith(401);
    expect(mockResponse.json).toHaveBeenCalledWith({
      error: {
        message: 'API key required',
        statusCode: 401,
      },
    });
  });

  it('should extract organization ID from headers', () => {
    const orgId = '123e4567-e89b-12d3-a456-426614174000';
    mockRequest.headers = {
      'x-api-key': 'test-api-key',
      'x-organization-id': orgId,
    };

    authMiddleware(
      mockRequest as AuthenticatedRequest,
      mockResponse as Response,
      nextFunction
    );

    expect(mockRequest.organizationId).toBe(orgId);
  });

  it('should handle errors gracefully', () => {
    mockRequest.headers = {
      'x-api-key': 'test-api-key',
    };
    
    // Force an error by making headers undefined
    const originalHeaders = mockRequest.headers;
    mockRequest.headers = undefined as any;

    authMiddleware(
      mockRequest as AuthenticatedRequest,
      mockResponse as Response,
      nextFunction
    );

    expect(mockResponse.status).toHaveBeenCalledWith(401);
    expect(mockResponse.json).toHaveBeenCalledWith({
      error: {
        message: 'Invalid API key',
        statusCode: 401,
      },
    });
  });
});
