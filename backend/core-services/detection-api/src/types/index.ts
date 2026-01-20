export interface DetectionRequest {
  emailContent?: string;
  url?: string;
  text?: string;
  image?: string; // base64 encoded
  legitimateDomain?: string;
  legitimateUrl?: string;
  organizationId?: string;
  includeFeatures?: boolean;
}

export interface MLServiceResponse {
  nlp: any | null;
  url: any | null;
  visual: any | null;
  processingTimeMs: number;
}

export interface NLPResponse {
  phishing_probability: number;
  legitimate_probability: number;
  confidence: number;
  prediction: string;
  ai_generated_probability?: number;
  urgency_score?: number;
  sentiment?: string;
  social_engineering_score?: number;
  features?: Record<string, any>;
  processing_time_ms?: number;
}

export interface URLResponse {
  domain_analysis?: {
    is_suspicious: boolean;
    score: number;
    [key: string]: any;
  };
  whois_analysis?: {
    is_suspicious: boolean;
    [key: string]: any;
  };
  redirect_analysis?: {
    is_suspicious: boolean;
    redirect_count: number;
    [key: string]: any;
  };
  homoglyph_analysis?: {
    is_suspicious: boolean;
    [key: string]: any;
  };
  phishing_probability?: number;
  [key: string]: any;
}

export interface VisualResponse {
  form_analysis?: {
    is_suspicious: boolean;
    [key: string]: any;
  };
  brand_prediction?: {
    is_brand_impersonation: boolean;
    [key: string]: any;
  };
  similarity_analysis?: {
    is_similar: boolean;
    similarity_score: number;
    [key: string]: any;
  };
  dom_analysis?: {
    text?: string;
    [key: string]: any;
  };
  phishing_probability?: number;
  [key: string]: any;
}
