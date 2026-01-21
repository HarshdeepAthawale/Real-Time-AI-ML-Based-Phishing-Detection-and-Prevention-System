import { IOCType } from '../models/ioc.model';

/**
 * Normalize IOC values for consistent lookups
 */

export function normalizeURL(url: string): string {
  try {
    const parsed = new URL(url);
    // Remove protocol, www, trailing slash, lowercase
    let normalized = parsed.hostname.replace(/^www\./i, '');
    normalized = normalized.toLowerCase().trim();
    
    // Add path if significant
    if (parsed.pathname && parsed.pathname !== '/') {
      normalized += parsed.pathname.toLowerCase().replace(/\/$/, '');
    }
    
    return normalized;
  } catch {
    // If URL parsing fails, just lowercase and trim
    return url.toLowerCase().trim().replace(/^https?:\/\//i, '');
  }
}

export function normalizeDomain(domain: string): string {
  // Remove www, lowercase, trim
  return domain
    .replace(/^www\./i, '')
    .toLowerCase()
    .trim();
}

export function normalizeIP(ip: string): string {
  // Remove whitespace, keep as-is for IPv4/IPv6
  return ip.trim();
}

export function normalizeHash(hash: string): string {
  // Uppercase, remove whitespace
  return hash.replace(/\s+/g, '').toUpperCase().trim();
}

export function normalizeEmail(email: string): string {
  // Lowercase, trim
  return email.toLowerCase().trim();
}

export function normalizeFilename(filename: string): string {
  // Lowercase, trim
  return filename.toLowerCase().trim();
}

/**
 * Normalize IOC value based on type
 */
export function normalizeIOCValue(iocType: IOCType, value: string): string {
  switch (iocType) {
    case 'url':
      return normalizeURL(value);
    case 'domain':
      return normalizeDomain(value);
    case 'ip':
      return normalizeIP(value);
    case 'hash_md5':
    case 'hash_sha1':
    case 'hash_sha256':
      return normalizeHash(value);
    case 'email':
      return normalizeEmail(value);
    case 'filename':
      return normalizeFilename(value);
    default:
      return value.toLowerCase().trim();
  }
}

/**
 * Generate hash for IOC value (for database indexing)
 */
export function hashIOCValue(value: string): string {
  const crypto = require('crypto');
  return crypto.createHash('sha256').update(value.toLowerCase().trim()).digest('hex');
}
