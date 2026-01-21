import { URL } from 'url';
import { logger } from '../utils/logger';

export class PrivacyFilterService {
  /**
   * Filter URL to remove path and query parameters for privacy
   * Returns only protocol and hostname
   */
  filterURL(url: string): string {
    try {
      const parsed = new URL(url);
      return `${parsed.protocol}//${parsed.hostname}`;
    } catch (error: any) {
      logger.warn('Failed to parse URL for privacy filtering', { url, error: error.message });
      // Return original URL if parsing fails
      return url;
    }
  }
  
  /**
   * Filter email content to extract minimal information
   * Returns only subject, from, and link metadata
   */
  filterEmailContent(emailContent: string): {
    subject: string;
    from: string;
    hasLinks: boolean;
    linkCount: number;
    links?: string[]; // Only domain parts for privacy
  } {
    try {
      // Extract subject
      const subjectMatch = emailContent.match(/Subject:\s*(.+)/i);
      const subject = subjectMatch ? subjectMatch[1].trim() : '';
      
      // Extract from
      const fromMatch = emailContent.match(/From:\s*(.+)/i);
      const from = fromMatch ? fromMatch[1].trim() : '';
      
      // Extract links (URLs)
      const linkPattern = /https?:\/\/[^\s<>"{}|\\^`\[\]]+/g;
      const links = emailContent.match(linkPattern) || [];
      
      // Extract only domains from links for privacy
      const linkDomains = links.map(link => {
        try {
          const url = new URL(link);
          return url.hostname;
        } catch {
          return '';
        }
      }).filter(domain => domain.length > 0);
      
      return {
        subject,
        from,
        hasLinks: links.length > 0,
        linkCount: links.length,
        links: linkDomains.length > 0 ? linkDomains : undefined
      };
    } catch (error: any) {
      logger.warn('Failed to filter email content', { error: error.message });
      return {
        subject: '',
        from: '',
        hasLinks: false,
        linkCount: 0
      };
    }
  }
  
  /**
   * Extract text content from email while preserving structure
   * Removes HTML tags and extracts plain text
   */
  extractTextContent(emailContent: string): string {
    try {
      // Remove HTML tags if present
      let text = emailContent.replace(/<[^>]*>/g, ' ');
      
      // Decode HTML entities
      text = text
        .replace(/&nbsp;/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'");
      
      // Clean up whitespace
      text = text.replace(/\s+/g, ' ').trim();
      
      return text.substring(0, 1000); // Limit to 1000 chars for privacy
    } catch (error: any) {
      logger.warn('Failed to extract text content', { error: error.message });
      // Handle null/undefined emailContent
      if (!emailContent || typeof emailContent !== 'string') {
        return '';
      }
      return emailContent.substring(0, 500); // Fallback: first 500 chars
    }
  }
}
