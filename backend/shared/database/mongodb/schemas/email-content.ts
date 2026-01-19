import { ObjectId } from 'mongodb';

export interface EmailContent {
  _id?: ObjectId;
  email_message_id: string; // Reference to PostgreSQL email_messages.id
  body_text: string;
  body_html: string;
  attachments?: Array<{
    filename: string;
    content_type: string;
    size: number;
    hash: string;
  }>;
  nlp_analysis?: {
    embeddings: number[];
    sentiment: string;
    urgency_score: number;
    ai_generated_probability: number;
    features: Record<string, any>;
  };
  created_at: Date;
  updated_at: Date;
}

// Index definitions
export const emailContentIndexes = [
  { email_message_id: 1 },
  { 'nlp_analysis.ai_generated_probability': -1 },
];
