import axios, { AxiosInstance } from 'axios';
import { IOC, IOCType, Severity } from '../models/ioc.model';
import { BaseFeedClient } from './base-feed.client';
import { logger } from '../utils/logger';

export class MISPClient extends BaseFeedClient {
  private client: AxiosInstance;
  private apiKey: string;
  private baseURL: string;
  
  constructor(baseURL: string, apiKey: string) {
    super('MISP');
    this.baseURL = baseURL;
    this.apiKey = apiKey;
    this.client = axios.create({
      baseURL: `${baseURL}/attributes/restSearch`,
      headers: {
        'Authorization': apiKey,
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      timeout: 30000, // 30 seconds
    });
  }
  
  async fetchIOCs(since?: Date): Promise<IOC[]> {
    try {
      const params: any = {
        returnFormat: 'json',
        type: ['url', 'domain', 'ip-dst', 'ip-src', 'md5', 'sha1', 'sha256', 'filename'],
      };
      
      if (since) {
        params.timestamp = Math.floor(since.getTime() / 1000);
      }
      
      const response = await this.retry(() => this.client.post('', params));
      const attributes = response.data.response?.Attribute || [];
      
      const iocs: IOC[] = attributes.map((attr: any) => this.mapToIOC(attr));
      
      logger.info(`Fetched ${iocs.length} IOCs from MISP`);
      return iocs;
    } catch (error) {
      logger.error('MISP fetch error', error);
      throw error;
    }
  }
  
  async publishIOC(ioc: IOC): Promise<void> {
    try {
      const event = {
        Event: {
          info: `Phishing detection: ${ioc.iocType}`,
          distribution: 1, // Your organization only
          threat_level_id: this.mapSeverityToThreatLevel(ioc.severity || 'low'),
          Attribute: [{
            type: this.mapIOCTypeToMISP(ioc.iocType),
            value: ioc.iocValue,
            category: 'Network activity',
            to_ids: true,
            comment: ioc.threatType || 'Phishing IOC',
          }]
        }
      };
      
      await this.retry(() => 
        axios.post(
          `${this.baseURL}/events/add`,
          event,
          {
            headers: {
              'Authorization': this.apiKey,
              'Content-Type': 'application/json',
              'Accept': 'application/json',
            },
            timeout: 30000,
          }
        )
      );
      
      logger.info(`Published IOC to MISP: ${ioc.iocValue}`);
    } catch (error) {
      logger.error('MISP publish error', error);
      throw error;
    }
  }
  
  private mapToIOC(attr: any): IOC {
    return {
      iocType: this.mapMISPTypeToIOC(attr.type),
      iocValue: attr.value,
      threatType: attr.Event?.info || 'unknown',
      severity: this.mapThreatLevelToSeverity(attr.Event?.threat_level_id),
      source: 'misp',
      firstSeenAt: attr.timestamp ? new Date(attr.timestamp * 1000) : new Date(),
      metadata: {
        eventId: attr.event_id,
        attributeId: attr.id,
        category: attr.category,
      }
    };
  }
  
  private mapIOCTypeToMISP(type: IOCType): string {
    const mapping: Record<IOCType, string> = {
      'url': 'url',
      'domain': 'domain',
      'ip': 'ip-dst',
      'hash_md5': 'md5',
      'hash_sha1': 'sha1',
      'hash_sha256': 'sha256',
      'filename': 'filename',
      'email': 'email-src',
    };
    return mapping[type] || type;
  }
  
  private mapMISPTypeToIOC(type: string): IOCType {
    const mapping: Record<string, IOCType> = {
      'url': 'url',
      'domain': 'domain',
      'ip-dst': 'ip',
      'ip-src': 'ip',
      'md5': 'hash_md5',
      'sha1': 'hash_sha1',
      'sha256': 'hash_sha256',
      'filename': 'filename',
      'email-src': 'email',
      'email-dst': 'email',
    };
    return mapping[type] || 'url';
  }
  
  private mapSeverityToThreatLevel(severity: Severity): number {
    const mapping: Record<Severity, number> = {
      'critical': 1,
      'high': 2,
      'medium': 3,
      'low': 4,
    };
    return mapping[severity] || 4;
  }
  
  private mapThreatLevelToSeverity(level: number): Severity {
    const mapping: Record<number, Severity> = {
      1: 'critical',
      2: 'high',
      3: 'medium',
      4: 'low',
    };
    return mapping[level] || 'low';
  }
}
