import axios, { AxiosInstance } from 'axios';
import { IOC, IOCType, Severity } from '../models/ioc.model';
import { BaseFeedClient } from './base-feed.client';
import { logger } from '../utils/logger';

interface OTXPulse {
  id: string;
  name: string;
  description: string;
  public: number;
  created: string;
  modified: string;
  TLP: string;
  tags: string[];
  indicators: OTXIndicator[];
}

interface OTXIndicator {
  id: string;
  indicator: string;
  type: string;
  title: string;
  description: string;
  content: string;
  created: string;
}

interface OTXResponse {
  count: number;
  next: string | null;
  previous: string | null;
  results: OTXPulse[];
}

export class OTXClient extends BaseFeedClient {
  private client: AxiosInstance;
  private apiKey: string;
  
  constructor(apiKey: string) {
    super('OTX');
    this.apiKey = apiKey;
    this.client = axios.create({
      baseURL: 'https://otx.alienvault.com/api/v1',
      headers: {
        'X-OTX-API-KEY': apiKey,
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 seconds
    });
  }
  
  async fetchIOCs(since?: Date): Promise<IOC[]> {
    try {
      const iocs: IOC[] = [];
      let nextUrl: string | null = '/pulses/subscribed';
      let page = 1;
      const maxPages = 100; // Limit pages to prevent infinite loops
      
      while (nextUrl && page <= maxPages) {
        const params: any = {
          limit: 100,
          page: page,
        };
        
        if (since) {
          params.modified_since = since.toISOString();
        }
        
        const response = await this.retry(() => {
          if (nextUrl?.startsWith('http')) {
            return axios.get(nextUrl, {
              headers: {
                'X-OTX-API-KEY': this.apiKey,
              },
            });
          }
          return this.client.get('/pulses/subscribed', { params });
        });
        
        const data: OTXResponse = response.data;
        const pulses = data.results || [];
        
        for (const pulse of pulses) {
          for (const indicator of pulse.indicators || []) {
            const ioc = this.mapToIOC(indicator, pulse);
            if (ioc) {
              iocs.push(ioc);
            }
          }
        }
        
        nextUrl = data.next;
        page++;
        
        // If no more pages, break
        if (!data.next || pulses.length === 0) {
          break;
        }
      }
      
      logger.info(`Fetched ${iocs.length} IOCs from OTX`);
      return iocs;
    } catch (error) {
      logger.error('OTX fetch error', error);
      throw error;
    }
  }
  
  async publishIOC(ioc: IOC): Promise<void> {
    // OTX doesn't have a direct API for publishing individual IOCs
    // This would require creating a pulse, which is more complex
    logger.warn('OTX publishIOC not implemented - requires pulse creation');
    throw new Error('OTX does not support direct IOC publishing via API');
  }
  
  private mapToIOC(indicator: OTXIndicator, pulse: OTXPulse): IOC | null {
    const iocType = this.mapOTXTypeToIOC(indicator.type);
    if (!iocType) {
      return null; // Unsupported type
    }
    
    return {
      iocType,
      iocValue: indicator.indicator,
      threatType: pulse.name,
      severity: this.mapTLPToSeverity(pulse.TLP),
      source: 'otx',
      firstSeenAt: indicator.created ? new Date(indicator.created) : new Date(),
      metadata: {
        pulseId: pulse.id,
        pulseName: pulse.name,
        tlp: pulse.TLP,
        tags: pulse.tags || [],
        description: pulse.description,
      }
    };
  }
  
  private mapOTXTypeToIOC(type: string): IOCType | null {
    const mapping: Record<string, IOCType> = {
      'URL': 'url',
      'domain': 'domain',
      'hostname': 'domain',
      'IPv4': 'ip',
      'IPv6': 'ip',
      'FileHash-MD5': 'hash_md5',
      'FileHash-SHA1': 'hash_sha1',
      'FileHash-SHA256': 'hash_sha256',
      'filename': 'filename',
      'email': 'email',
    };
    
    return mapping[type] || null;
  }
  
  private mapTLPToSeverity(tlp: string): Severity {
    const mapping: Record<string, Severity> = {
      'RED': 'critical',
      'AMBER': 'high',
      'GREEN': 'medium',
      'WHITE': 'low',
    };
    return mapping[tlp] || 'low';
  }
}
