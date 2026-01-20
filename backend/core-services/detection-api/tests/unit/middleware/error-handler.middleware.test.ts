import { Request, Response, NextFunction } from 'express';
import { errorHandler, AppError } from '../../../src/middleware/error-handler.middleware';

jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('errorHandler', () => {
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;
  let nextFunction: NextFunction;

  beforeEach(() => {
    mockRequest = {
      path: '/api/v1/detect/email',
      method: 'POST',
    };

    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    };

    nextFunction = jest.fn();
  });

  it('should handle errors with status code', () => {
    const error: AppError = new Error('Test error');
    error.statusCode = 400;

    errorHandler(
      error,
      mockRequest as Request,
      mockResponse as Response,
      nextFunction
    );

    expect(mockResponse.status).toHaveBeenCalledWith(400);
    expect(mockResponse.json).toHaveBeenCalledWith({
      error: {
        message: 'Test error',
        statusCode: 400,
        timestamp: expect.any(String),
        path: '/api/v1/detect/email',
      },
    });
  });

  it('should default to 500 for errors without status code', () => {
    const error = new Error('Internal server error');

    errorHandler(
      error,
      mockRequest as Request,
      mockResponse as Response,
      nextFunction
    );

    expect(mockResponse.status).toHaveBeenCalledWith(500);
    expect(mockResponse.json).toHaveBeenCalledWith({
      error: {
        message: 'Internal server error',
        statusCode: 500,
        timestamp: expect.any(String),
        path: '/api/v1/detect/email',
      },
    });
  });

  it('should use default message when error has no message', () => {
    const error = new Error();
    error.message = '';

    errorHandler(
      error,
      mockRequest as Request,
      mockResponse as Response,
      nextFunction
    );

    expect(mockResponse.json).toHaveBeenCalledWith(
      expect.objectContaining({
        error: expect.objectContaining({
          message: 'Internal Server Error',
        }),
      })
    );
  });

  it('should include timestamp in error response', () => {
    const error = new Error('Test error');

    errorHandler(
      error,
      mockRequest as Request,
      mockResponse as Response,
      nextFunction
    );

    const jsonCall = (mockResponse.json as jest.Mock).mock.calls[0][0];
    expect(jsonCall.error.timestamp).toBeDefined();
    expect(new Date(jsonCall.error.timestamp).getTime()).toBeLessThanOrEqual(
      Date.now()
    );
  });

  it('should include path in error response', () => {
    const error = new Error('Test error');

    errorHandler(
      error,
      mockRequest as Request,
      mockResponse as Response,
      nextFunction
    );

    const jsonCall = (mockResponse.json as jest.Mock).mock.calls[0][0];
    expect(jsonCall.error.path).toBe('/api/v1/detect/email');
  });
});
