import { detectEmailSchema, detectURLSchema, detectTextSchema } from '../../../src/utils/validators';

describe('Validators', () => {
  describe('detectEmailSchema', () => {
    it('should validate valid email content', () => {
      const valid = {
        emailContent: 'This is a test email content',
      };

      expect(() => detectEmailSchema.parse(valid)).not.toThrow();
    });

    it('should validate email with organization ID', () => {
      const valid = {
        emailContent: 'Test email',
        organizationId: '123e4567-e89b-12d3-a456-426614174000',
      };

      expect(() => detectEmailSchema.parse(valid)).not.toThrow();
    });

    it('should validate email with includeFeatures', () => {
      const valid = {
        emailContent: 'Test email',
        includeFeatures: true,
      };

      expect(() => detectEmailSchema.parse(valid)).not.toThrow();
    });

    it('should reject empty email content', () => {
      const invalid = {
        emailContent: '',
      };

      expect(() => detectEmailSchema.parse(invalid)).toThrow();
    });

    it('should reject missing email content', () => {
      const invalid = {};

      expect(() => detectEmailSchema.parse(invalid)).toThrow();
    });

    it('should reject invalid UUID for organizationId', () => {
      const invalid = {
        emailContent: 'Test email',
        organizationId: 'invalid-uuid',
      };

      expect(() => detectEmailSchema.parse(invalid)).toThrow();
    });
  });

  describe('detectURLSchema', () => {
    it('should validate valid URL', () => {
      const valid = {
        url: 'https://example.com',
      };

      expect(() => detectURLSchema.parse(valid)).not.toThrow();
    });

    it('should validate URL with legitimate domain', () => {
      const valid = {
        url: 'https://example.com',
        legitimateDomain: 'example.com',
      };

      expect(() => detectURLSchema.parse(valid)).not.toThrow();
    });

    it('should validate URL with legitimate URL', () => {
      const valid = {
        url: 'https://example.com',
        legitimateUrl: 'https://legitimate.com',
      };

      expect(() => detectURLSchema.parse(valid)).not.toThrow();
    });

    it('should reject invalid URL format', () => {
      const invalid = {
        url: 'not-a-url',
      };

      expect(() => detectURLSchema.parse(invalid)).toThrow();
    });

    it('should reject missing URL', () => {
      const invalid = {};

      expect(() => detectURLSchema.parse(invalid)).toThrow();
    });

    it('should reject invalid legitimate URL format', () => {
      const invalid = {
        url: 'https://example.com',
        legitimateUrl: 'not-a-url',
      };

      expect(() => detectURLSchema.parse(invalid)).toThrow();
    });
  });

  describe('detectTextSchema', () => {
    it('should validate valid text', () => {
      const valid = {
        text: 'This is test text content',
      };

      expect(() => detectTextSchema.parse(valid)).not.toThrow();
    });

    it('should validate text with includeFeatures', () => {
      const valid = {
        text: 'Test text',
        includeFeatures: false,
      };

      expect(() => detectTextSchema.parse(valid)).not.toThrow();
    });

    it('should validate text with organization ID', () => {
      const valid = {
        text: 'Test text',
        organizationId: '123e4567-e89b-12d3-a456-426614174000',
      };

      expect(() => detectTextSchema.parse(valid)).not.toThrow();
    });

    it('should reject empty text', () => {
      const invalid = {
        text: '',
      };

      expect(() => detectTextSchema.parse(invalid)).toThrow();
    });

    it('should reject missing text', () => {
      const invalid = {};

      expect(() => detectTextSchema.parse(invalid)).toThrow();
    });
  });
});
