import { IOC } from '../models/ioc.model';
import { logger } from '../utils/logger';

/**
 * Base interface for threat intelligence feed clients
 */
export interface IFeedClient {
  fetchIOCs(since?: Date): Promise<IOC[]>;
  publishIOC(ioc: IOC): Promise<void>;
  getFeedName(): string;
}

/**
 * Base class for feed clients with common functionality
 */
export abstract class BaseFeedClient implements IFeedClient {
  protected feedName: string;
  
  constructor(feedName: string) {
    this.feedName = feedName;
  }
  
  abstract fetchIOCs(since?: Date): Promise<IOC[]>;
  abstract publishIOC(ioc: IOC): Promise<void>;
  
  getFeedName(): string {
    return this.feedName;
  }
  
  /**
   * Retry helper for API calls
   */
  protected async retry<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    delay: number = 1000
  ): Promise<T> {
    let lastError: Error;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        if (i < maxRetries - 1) {
          logger.warn(`Retry ${i + 1}/${maxRetries} for ${this.feedName}`, error);
          await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
        }
      }
    }
    
    throw lastError!;
  }
}
