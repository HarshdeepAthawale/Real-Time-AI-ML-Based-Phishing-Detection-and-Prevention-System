/**
 * Redis Cache Key Patterns
 * 
 * URL reputation cache
 * Key: `url:reputation:{url_hash}`
 * Value: JSON string with reputation score and metadata
 * TTL: 1 hour
 */
export const getURLReputationKey = (urlHash: string): string => {
  return `url:reputation:${urlHash}`;
};

/**
 * Domain reputation cache
 * Key: `domain:reputation:{domain}`
 * Value: JSON string with reputation score
 * TTL: 6 hours
 */
export const getDomainReputationKey = (domain: string): string => {
  return `domain:reputation:${domain}`;
};

/**
 * IOC lookup cache (Bloom filter for fast negative lookups)
 * Key: `ioc:bloom:{ioc_type}`
 * Use RedisBloom module
 */
export const getIOCBloomKey = (iocType: string): string => {
  return `ioc:bloom:${iocType}`;
};

/**
 * Model inference cache
 * Key: `model:inference:{model_type}:{input_hash}`
 * Value: JSON string with model output
 * TTL: 24 hours
 */
export const getModelInferenceKey = (modelType: string, inputHash: string): string => {
  return `model:inference:${modelType}:${inputHash}`;
};

// Cache TTLs (in seconds)
export const CACHE_TTL = {
  URL_REPUTATION: 3600, // 1 hour
  DOMAIN_REPUTATION: 21600, // 6 hours
  MODEL_INFERENCE: 86400, // 24 hours
};
