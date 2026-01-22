// API Response Types

export interface ApiError {
  error: string;
  message?: string;
  details?: any;
  statusCode?: number;
}

export interface ApiResponse<T> {
  data?: T;
  error?: ApiError;
}

// Dashboard Types
export interface DashboardStats {
  criticalThreats: number;
  detectionRate: number;
  avgResponseTime: number;
  phishingAttempts: number;
  criticalThreatsChange?: string;
  detectionRateChange?: string;
  avgResponseTimeChange?: string;
  phishingAttemptsPeriod?: string;
}

export interface Threat {
  id: string;
  type: string;
  target: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'blocked' | 'monitored' | 'resolved' | 'detected';
  timestamp: string;
  confidenceScore?: number;
  source?: string;
  title?: string;
  description?: string;
}

export interface ChartDataPoint {
  time: string;
  threats: number;
}

export interface ThreatDistribution {
  emailPhishing: number;
  urlSpoofing: number;
  domainHijacking: number;
  aiGeneratedContent: number;
}

// Threat Intelligence Types
export interface MaliciousDomain {
  domain: string;
  reputation: 'Malicious' | 'Suspicious' | 'Clean';
  reports: number;
  firstSeen?: string;
  lastSeen?: string;
}

export interface ThreatPattern {
  pattern: string;
  incidents: number;
  severity?: 'critical' | 'high' | 'medium' | 'low';
}

export interface IOC {
  value: string;
  type: 'IP Address' | 'File Hash' | 'Domain' | 'Filename' | 'URL' | 'Email';
  sources: number;
  firstSeen?: string;
  lastSeen?: string;
  severity?: 'critical' | 'high' | 'medium' | 'low';
}

export interface ThreatIntelligenceSummary {
  knownThreats: number;
  feedIntegrations: number;
  zeroDayDetection: number;
  lastUpdated: string;
}

// Detection Types
export interface DetectionRequest {
  emailContent?: string;
  url?: string;
  text?: string;
  image?: string;
  legitimateDomain?: string;
  legitimateUrl?: string;
  organizationId?: string;
  includeFeatures?: boolean;
}

export interface DetectionResult {
  isThreat: boolean;
  confidence: number;
  severity: 'critical' | 'high' | 'medium' | 'low';
  threatType: string;
  scores: {
    ensemble: number;
    nlp: number;
    url: number;
    visual: number;
  };
  indicators: string[];
  metadata: {
    processingTimeMs: number;
    timestamp: string;
  };
  cached?: boolean;
}

// WebSocket Event Types
export interface LiveEvent {
  id: string;
  type: 'detection' | 'blocked' | 'alert' | 'threat_detected' | 'url_analyzed' | 'email_analyzed';
  message: string;
  timestamp: Date;
  data?: any;
}

export interface WebSocketEvent {
  event: string;
  data: any;
  timestamp: string;
}
