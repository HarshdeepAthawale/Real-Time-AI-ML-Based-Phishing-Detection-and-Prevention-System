import { MLServiceResponse, NLPResponse, URLResponse, VisualResponse } from '../../src/types';
import { Threat } from '../../src/models/detection.model';

export const mockNLPResponse: NLPResponse = {
  phishing_probability: 0.85,
  legitimate_probability: 0.15,
  confidence: 0.70,
  prediction: 'phishing',
  ai_generated_probability: 0.75,
  urgency_score: 80.0,
  sentiment: 'NEGATIVE',
  social_engineering_score: 70.0,
  processing_time_ms: 45.2,
};

export const mockNLPResponseLow: NLPResponse = {
  phishing_probability: 0.25,
  legitimate_probability: 0.75,
  confidence: 0.60,
  prediction: 'legitimate',
  ai_generated_probability: 0.20,
  urgency_score: 30.0,
  sentiment: 'NEUTRAL',
  social_engineering_score: 20.0,
  processing_time_ms: 40.0,
};

export const mockURLResponse: URLResponse = {
  domain_analysis: {
    is_suspicious: true,
    score: 0.8,
    tld: '.com',
    subdomain_count: 2,
  },
  whois_analysis: {
    is_suspicious: true,
    age_days: 5,
    registrar: 'Unknown',
  },
  redirect_analysis: {
    is_suspicious: true,
    redirect_count: 4,
    final_url: 'https://malicious-site.com',
  },
  homoglyph_analysis: {
    is_suspicious: true,
    has_homoglyph: true,
    original_domain: 'paypal.com',
    detected_domain: 'paypai.com',
  },
  phishing_probability: 0.90,
};

export const mockURLResponseLegitimate: URLResponse = {
  domain_analysis: {
    is_suspicious: false,
    score: 0.1,
    tld: '.com',
    subdomain_count: 1,
  },
  whois_analysis: {
    is_suspicious: false,
    age_days: 3650,
    registrar: 'GoDaddy',
  },
  redirect_analysis: {
    is_suspicious: false,
    redirect_count: 0,
    final_url: null,
  },
  homoglyph_analysis: {
    is_suspicious: false,
    has_homoglyph: false,
  },
  phishing_probability: 0.05,
};

export const mockVisualResponse: VisualResponse = {
  form_analysis: {
    is_suspicious: true,
    form_count: 2,
    password_fields: 1,
    suspicious_fields: ['ssn', 'credit_card'],
  },
  brand_prediction: {
    is_brand_impersonation: true,
    predicted_brand: 'PayPal',
    confidence: 0.85,
  },
  similarity_analysis: {
    is_similar: true,
    similarity_score: 0.75,
    hamming_distance: 15,
  },
  dom_analysis: {
    text: 'Please enter your PayPal credentials to verify your account.',
    link_count: 5,
  },
  phishing_probability: 0.80,
};

export const mockVisualResponseLegitimate: VisualResponse = {
  form_analysis: {
    is_suspicious: false,
    form_count: 1,
    password_fields: 1,
    suspicious_fields: [],
  },
  brand_prediction: {
    is_brand_impersonation: false,
    predicted_brand: null,
    confidence: 0.1,
  },
  similarity_analysis: {
    is_similar: false,
    similarity_score: 0.95,
    hamming_distance: 2,
  },
  dom_analysis: {
    text: 'Welcome to PayPal. Sign in to your account.',
    link_count: 10,
  },
  phishing_probability: 0.10,
};

export const mockMLServiceResponse: MLServiceResponse = {
  nlp: mockNLPResponse,
  url: mockURLResponse,
  visual: mockVisualResponse,
  processingTimeMs: 150,
};

export const mockMLServiceResponseEmail: MLServiceResponse = {
  nlp: mockNLPResponse,
  url: [mockURLResponse],
  visual: null,
  processingTimeMs: 120,
};

export const mockMLServiceResponseURL: MLServiceResponse = {
  nlp: mockNLPResponse,
  url: mockURLResponse,
  visual: mockVisualResponse,
  processingTimeMs: 200,
};

export const mockThreat: Threat = {
  isThreat: true,
  confidence: 0.85,
  severity: 'high',
  threatType: 'email_phishing',
  scores: {
    ensemble: 0.85,
    nlp: 0.85,
    url: 0.90,
    visual: 0.80,
  },
  indicators: [
    'high_urgency_language',
    'ai_generated_content',
    'homoglyph_attack',
    'excessive_redirects',
    'credential_harvesting_form',
    'brand_impersonation',
  ],
  metadata: {
    processingTimeMs: 150,
    timestamp: new Date().toISOString(),
  },
};

export const mockThreatLow: Threat = {
  isThreat: false,
  confidence: 0.30,
  severity: 'low',
  threatType: 'email_phishing',
  scores: {
    ensemble: 0.30,
    nlp: 0.25,
    url: 0.05,
    visual: 0.10,
  },
  indicators: [],
  metadata: {
    processingTimeMs: 100,
    timestamp: new Date().toISOString(),
  },
};
