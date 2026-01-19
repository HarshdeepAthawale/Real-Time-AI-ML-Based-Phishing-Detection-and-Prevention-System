import { ObjectId } from 'mongodb';

export interface URLAnalysis {
  _id?: ObjectId;
  url_id: string; // Reference to PostgreSQL urls.id
  domain_id: string; // Reference to PostgreSQL domains.id
  graph_analysis?: {
    node_embeddings: number[];
    cluster_id: string;
    relationships: Array<{
      related_domain_id: string;
      relationship_type: string;
      strength: number;
    }>;
  };
  gnn_analysis?: {
    malicious_probability: number;
    anomaly_score: number;
    features: Record<string, any>;
  };
  redirect_chain_analysis?: {
    hops: Array<{
      url: string;
      status_code: number;
      redirect_type: string;
    }>;
    suspicious_patterns: string[];
  };
  created_at: Date;
  updated_at: Date;
}

// Index definitions
export const urlAnalysisIndexes = [
  { url_id: 1 },
  { domain_id: 1 },
  { 'gnn_analysis.malicious_probability': -1 },
];
