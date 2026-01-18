// Shared TypeScript type definitions

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    message: string;
    code?: string;
    details?: any;
  };
  timestamp: string;
}

export interface DetectionRequest {
  content: string;
  contentType: 'email' | 'url' | 'text';
  metadata?: Record<string, any>;
}

export interface DetectionResult {
  isPhishing: boolean;
  confidence: number;
  riskScore: number;
  reasons: string[];
  modelVersion: string;
  timestamp: string;
}

export interface ApiKey {
  id: string;
  key: string;
  name: string;
  userId: string;
  rateLimit: number;
  createdAt: string;
  expiresAt?: string;
  isActive: boolean;
}

export interface ServiceHealth {
  service: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  responseTime?: number;
  version?: string;
}

export interface LogEntry {
  level: 'info' | 'warn' | 'error' | 'debug';
  message: string;
  service: string;
  timestamp: string;
  metadata?: Record<string, any>;
}
