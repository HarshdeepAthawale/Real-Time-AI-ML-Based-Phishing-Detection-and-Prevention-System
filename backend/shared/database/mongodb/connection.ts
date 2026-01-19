import { MongoClient, Db, Collection } from 'mongodb';
import { config } from '../../config';
import {
  EmailContent,
  URLAnalysis,
  VisualAnalysis,
} from './schemas';

let client: MongoClient | null = null;
let db: Db | null = null;

export const connectMongoDB = async (): Promise<Db> => {
  if (db) {
    return db;
  }

  try {
    client = new MongoClient(config.database.mongodbUrl);
    await client.connect();
    db = client.db();
    
    // Create indexes
    await createIndexes();
    
    return db;
  } catch (error) {
    console.error('Failed to connect to MongoDB:', error);
    throw error;
  }
};

export const getMongoDB = (): Db => {
  if (!db) {
    throw new Error('MongoDB not connected. Call connectMongoDB() first.');
  }
  return db;
};

export const disconnectMongoDB = async (): Promise<void> => {
  if (client) {
    await client.close();
    client = null;
    db = null;
  }
};

const createIndexes = async (): Promise<void> => {
  if (!db) return;

  try {
    // Email content indexes
    const emailContentCollection = db.collection<EmailContent>('email_content');
    await emailContentCollection.createIndex({ email_message_id: 1 }, { unique: true });
    await emailContentCollection.createIndex({ 'nlp_analysis.ai_generated_probability': -1 });

    // URL analysis indexes
    const urlAnalysisCollection = db.collection<URLAnalysis>('url_analysis');
    await urlAnalysisCollection.createIndex({ url_id: 1 }, { unique: true });
    await urlAnalysisCollection.createIndex({ domain_id: 1 });
    await urlAnalysisCollection.createIndex({ 'gnn_analysis.malicious_probability': -1 });

    // Visual analysis indexes
    const visualAnalysisCollection = db.collection<VisualAnalysis>('visual_analysis');
    await visualAnalysisCollection.createIndex({ url_id: 1 }, { unique: true });
    await visualAnalysisCollection.createIndex({ 'cnn_analysis.brand_impersonation_score': -1 });
  } catch (error) {
    console.error('Failed to create MongoDB indexes:', error);
  }
};

// Collection getters
export const getEmailContentCollection = (): Collection<EmailContent> => {
  return getMongoDB().collection<EmailContent>('email_content');
};

export const getURLAnalysisCollection = (): Collection<URLAnalysis> => {
  return getMongoDB().collection<URLAnalysis>('url_analysis');
};

export const getVisualAnalysisCollection = (): Collection<VisualAnalysis> => {
  return getMongoDB().collection<VisualAnalysis>('visual_analysis');
};
