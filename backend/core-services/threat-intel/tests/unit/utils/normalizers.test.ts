import {
  normalizeURL,
  normalizeDomain,
  normalizeIP,
  normalizeHash,
  normalizeEmail,
  normalizeIOCValue,
  hashIOCValue,
} from '../../../src/utils/normalizers';

describe('IOC Normalizers', () => {
  describe('normalizeURL', () => {
    it('should normalize URL correctly', () => {
      expect(normalizeURL('https://www.example.com/path')).toBe('example.com/path');
      expect(normalizeURL('http://example.com/')).toBe('example.com');
      expect(normalizeURL('HTTPS://EXAMPLE.COM')).toBe('example.com');
    });
  });

  describe('normalizeDomain', () => {
    it('should normalize domain correctly', () => {
      expect(normalizeDomain('www.example.com')).toBe('example.com');
      expect(normalizeDomain('EXAMPLE.COM')).toBe('example.com');
      expect(normalizeDomain('  example.com  ')).toBe('example.com');
    });
  });

  describe('normalizeIP', () => {
    it('should normalize IP correctly', () => {
      expect(normalizeIP('192.168.1.1')).toBe('192.168.1.1');
      expect(normalizeIP('  192.168.1.1  ')).toBe('192.168.1.1');
      expect(normalizeIP('2001:0db8:85a3:0000:0000:8a2e:0370:7334')).toBe('2001:0db8:85a3:0000:0000:8a2e:0370:7334');
    });
  });

  describe('normalizeHash', () => {
    it('should normalize hash correctly', () => {
      expect(normalizeHash('abc123')).toBe('ABC123');
      expect(normalizeHash('abc 123')).toBe('ABC123');
      expect(normalizeHash('  abc123  ')).toBe('ABC123');
    });
  });

  describe('normalizeEmail', () => {
    it('should normalize email correctly', () => {
      expect(normalizeEmail('Test@Example.COM')).toBe('test@example.com');
      expect(normalizeEmail('  test@example.com  ')).toBe('test@example.com');
    });
  });

  describe('normalizeIOCValue', () => {
    it('should normalize based on IOC type', () => {
      expect(normalizeIOCValue('url', 'https://www.example.com')).toBe('example.com');
      expect(normalizeIOCValue('domain', 'WWW.EXAMPLE.COM')).toBe('example.com');
      expect(normalizeIOCValue('ip', '192.168.1.1')).toBe('192.168.1.1');
      expect(normalizeIOCValue('hash_md5', 'abc123')).toBe('ABC123');
      expect(normalizeIOCValue('email', 'Test@Example.COM')).toBe('test@example.com');
    });
  });

  describe('hashIOCValue', () => {
    it('should generate consistent hash', () => {
      const hash1 = hashIOCValue('example.com');
      const hash2 = hashIOCValue('example.com');
      expect(hash1).toBe(hash2);
      expect(hash1).toHaveLength(64); // SHA256 produces 64 hex characters
    });

    it('should be case-insensitive', () => {
      const hash1 = hashIOCValue('EXAMPLE.COM');
      const hash2 = hashIOCValue('example.com');
      expect(hash1).toBe(hash2);
    });
  });
});
