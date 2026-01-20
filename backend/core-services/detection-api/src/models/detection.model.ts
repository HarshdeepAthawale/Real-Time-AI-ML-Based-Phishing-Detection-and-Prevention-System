export type ThreatSeverity = 'low' | 'medium' | 'high' | 'critical';
export type ThreatType = 'email_phishing' | 'url_spoofing' | 'brand_impersonation' | 'ai_generated';

export interface Threat {
  isThreat: boolean;
  confidence: number;
  severity: ThreatSeverity;
  threatType: ThreatType;
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
}

export interface DetectionResult extends Threat {
  cached?: boolean;
  requestId?: string;
}
