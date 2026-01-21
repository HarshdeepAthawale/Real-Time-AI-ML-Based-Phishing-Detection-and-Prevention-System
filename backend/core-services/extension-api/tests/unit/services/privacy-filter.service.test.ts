import { PrivacyFilterService } from '../../../src/services/privacy-filter.service';

describe('PrivacyFilterService', () => {
  let privacyFilter: PrivacyFilterService;

  beforeEach(() => {
    privacyFilter = new PrivacyFilterService();
  });

  describe('filterURL', () => {
    it('should return only protocol and hostname', () => {
      const url = 'https://example.com/path/to/page?query=value&other=param';
      const result = privacyFilter.filterURL(url);
      expect(result).toBe('https://example.com');
    });

    it('should handle URLs without path', () => {
      const url = 'https://example.com';
      const result = privacyFilter.filterURL(url);
      expect(result).toBe('https://example.com');
    });

    it('should handle URLs with port', () => {
      const url = 'https://example.com:8080/path';
      const result = privacyFilter.filterURL(url);
      // Privacy filter strips port for privacy - only returns protocol and hostname
      expect(result).toBe('https://example.com');
    });

    it('should return original URL if parsing fails', () => {
      const invalidUrl = 'not-a-valid-url';
      const result = privacyFilter.filterURL(invalidUrl);
      expect(result).toBe(invalidUrl);
    });

    it('should handle http URLs', () => {
      const url = 'http://example.com/path';
      const result = privacyFilter.filterURL(url);
      expect(result).toBe('http://example.com');
    });
  });

  describe('filterEmailContent', () => {
    it('should extract subject and from', () => {
      const emailContent = 'Subject: Test Email\nFrom: sender@example.com\nBody: Content here';
      const result = privacyFilter.filterEmailContent(emailContent);

      expect(result.subject).toBe('Test Email');
      expect(result.from).toBe('sender@example.com');
    });

    it('should detect and count links', () => {
      const emailContent = 'Check out https://example.com and https://test.com';
      const result = privacyFilter.filterEmailContent(emailContent);

      expect(result.hasLinks).toBe(true);
      expect(result.linkCount).toBe(2);
      expect(result.links).toEqual(['example.com', 'test.com']);
    });

    it('should extract only domains from links', () => {
      const emailContent = 'Visit https://example.com/path?query=value';
      const result = privacyFilter.filterEmailContent(emailContent);

      expect(result.links).toEqual(['example.com']);
    });

    it('should handle emails without subject', () => {
      const emailContent = 'From: sender@example.com\nBody: Content';
      const result = privacyFilter.filterEmailContent(emailContent);

      expect(result.subject).toBe('');
      expect(result.from).toBe('sender@example.com');
    });

    it('should handle emails without from', () => {
      const emailContent = 'Subject: Test\nBody: Content';
      const result = privacyFilter.filterEmailContent(emailContent);

      expect(result.subject).toBe('Test');
      expect(result.from).toBe('');
    });

    it('should handle emails without links', () => {
      const emailContent = 'Subject: Test\nFrom: sender@example.com\nBody: No links here';
      const result = privacyFilter.filterEmailContent(emailContent);

      expect(result.hasLinks).toBe(false);
      expect(result.linkCount).toBe(0);
      expect(result.links).toBeUndefined();
    });

    it('should filter out invalid URLs from links', () => {
      const emailContent = 'Check https://valid.com and invalid-url';
      const result = privacyFilter.filterEmailContent(emailContent);

      expect(result.links).toEqual(['valid.com']);
    });

    it('should handle case-insensitive headers', () => {
      const emailContent = 'SUBJECT: Test\nFROM: sender@example.com';
      const result = privacyFilter.filterEmailContent(emailContent);

      expect(result.subject).toBe('Test');
      expect(result.from).toBe('sender@example.com');
    });

    it('should return empty values on parsing error', () => {
      const invalidContent = null as any;
      const result = privacyFilter.filterEmailContent(invalidContent);

      expect(result.subject).toBe('');
      expect(result.from).toBe('');
      expect(result.hasLinks).toBe(false);
      expect(result.linkCount).toBe(0);
    });
  });

  describe('extractTextContent', () => {
    it('should remove HTML tags', () => {
      const htmlContent = '<html><body><p>Hello</p><div>World</div></body></html>';
      const result = privacyFilter.extractTextContent(htmlContent);
      expect(result).toBe('Hello World');
    });

    it('should decode HTML entities', () => {
      const htmlContent = 'Hello &amp; World &lt;test&gt;';
      const result = privacyFilter.extractTextContent(htmlContent);
      expect(result).toContain('Hello & World');
      expect(result).toContain('<test>');
    });

    it('should clean up whitespace', () => {
      const content = 'Hello    World\n\nTest';
      const result = privacyFilter.extractTextContent(content);
      expect(result).toBe('Hello World Test');
    });

    it('should limit to 1000 characters', () => {
      const longContent = 'a'.repeat(2000);
      const result = privacyFilter.extractTextContent(longContent);
      expect(result.length).toBe(1000);
    });

    it('should handle plain text', () => {
      const textContent = 'This is plain text content';
      const result = privacyFilter.extractTextContent(textContent);
      expect(result).toBe('This is plain text content');
    });

    it('should return first 500 chars on error', () => {
      const invalidContent = null as any;
      const result = privacyFilter.extractTextContent(invalidContent);
      expect(result).toBe('');
    });

    it('should handle empty string', () => {
      const result = privacyFilter.extractTextContent('');
      expect(result).toBe('');
    });

    it('should handle mixed HTML and text', () => {
      const content = '<p>Hello</p>World<div>Test</div>';
      const result = privacyFilter.extractTextContent(content);
      expect(result).toBe('Hello World Test');
    });
  });
});
