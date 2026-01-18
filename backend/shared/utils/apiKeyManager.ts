// API Key Management Utilities

import bcrypt from 'bcrypt';
import crypto from 'crypto';

export interface ApiKeyData {
  id: string;
  keyHash: string;
  name: string;
  userId: string;
  rateLimit: number;
  createdAt: Date;
  expiresAt?: Date;
  isActive: boolean;
}

/**
 * Generate a new API key
 */
export const generateApiKey = (): string => {
  return `pd_${crypto.randomBytes(32).toString('hex')}`;
};

/**
 * Hash an API key using bcrypt
 */
export const hashApiKey = async (apiKey: string): Promise<string> => {
  const saltRounds = 10;
  return bcrypt.hash(apiKey, saltRounds);
};

/**
 * Verify an API key against a hash
 */
export const verifyApiKey = async (
  apiKey: string,
  hash: string
): Promise<boolean> => {
  return bcrypt.compare(apiKey, hash);
};

/**
 * Check if an API key is expired
 */
export const isApiKeyExpired = (expiresAt?: Date): boolean => {
  if (!expiresAt) {
    return false;
  }
  return new Date() > expiresAt;
};

/**
 * Check if an API key is active
 */
export const isApiKeyActive = (apiKeyData: ApiKeyData): boolean => {
  return apiKeyData.isActive && !isApiKeyExpired(apiKeyData.expiresAt);
};

/**
 * Create API key data structure
 */
export const createApiKeyData = async (
  name: string,
  userId: string,
  rateLimit: number = 100,
  expiresInDays?: number
): Promise<{ apiKey: string; apiKeyData: Omit<ApiKeyData, 'keyHash'> & { keyHash: string } }> => {
  const apiKey = generateApiKey();
  const keyHash = await hashApiKey(apiKey);
  const id = crypto.randomUUID();
  const createdAt = new Date();
  const expiresAt = expiresInDays
    ? new Date(createdAt.getTime() + expiresInDays * 24 * 60 * 60 * 1000)
    : undefined;

  const apiKeyData: Omit<ApiKeyData, 'keyHash'> & { keyHash: string } = {
    id,
    keyHash,
    name,
    userId,
    rateLimit,
    createdAt,
    expiresAt,
    isActive: true,
  };

  return { apiKey, apiKeyData };
};
