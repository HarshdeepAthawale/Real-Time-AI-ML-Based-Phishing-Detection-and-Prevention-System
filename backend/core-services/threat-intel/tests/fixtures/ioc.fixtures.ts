import { IOC } from '../../src/models/ioc.model';

export const mockIOC: IOC = {
  iocType: 'domain',
  iocValue: 'malicious.example.com',
  threatType: 'phishing',
  severity: 'high',
  confidence: 90,
  source: 'feed',
  metadata: {},
};

export const mockIOCs: IOC[] = [
  {
    iocType: 'domain',
    iocValue: 'malicious1.example.com',
    threatType: 'phishing',
    severity: 'high',
    confidence: 90,
    source: 'feed',
  },
  {
    iocType: 'ip',
    iocValue: '192.168.1.100',
    threatType: 'malware',
    severity: 'critical',
    confidence: 95,
    source: 'feed',
  },
  {
    iocType: 'url',
    iocValue: 'https://phishing.example.com/login',
    threatType: 'phishing',
    severity: 'medium',
    confidence: 80,
    source: 'user',
  },
];
