import { ObjectId } from 'mongodb';

export interface VisualAnalysis {
  _id?: ObjectId;
  url_id: string;
  screenshot_s3_path: string;
  dom_structure: {
    tree_hash: string;
    element_count: number;
    form_fields: Array<{
      type: string;
      name: string;
      placeholder?: string;
    }>;
    links: Array<{
      href: string;
      text: string;
    }>;
  };
  cnn_analysis?: {
    brand_impersonation_score: number;
    visual_similarity_scores: Array<{
      legitimate_domain: string;
      similarity: number;
    }>;
    features: Record<string, any>;
  };
  created_at: Date;
  updated_at: Date;
}

// Index definitions
export const visualAnalysisIndexes = [
  { url_id: 1 },
  { 'cnn_analysis.brand_impersonation_score': -1 },
];
