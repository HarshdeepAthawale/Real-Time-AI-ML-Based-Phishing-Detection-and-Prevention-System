import { apiGet, apiPost, apiPut, apiDelete } from '../api-client';

export interface ThreatFeed {
  id: string;
  name: string;
  feedType: 'misp' | 'otx' | 'custom' | 'user_submitted';
  apiEndpoint?: string;
  apiKeyEncrypted?: string;
  syncIntervalMinutes?: number;
  isActive: boolean;
  lastSyncAt?: string;
  createdAt: string;
  updatedAt: string;
}

export interface CreateFeedRequest {
  name: string;
  feedType: 'misp' | 'otx' | 'custom' | 'user_submitted';
  apiEndpoint?: string;
  apiKeyEncrypted?: string;
  syncIntervalMinutes?: number;
  isActive?: boolean;
}

export interface UpdateFeedRequest extends Partial<CreateFeedRequest> {}

export interface FeedListResponse {
  feeds: ThreatFeed[];
}

/**
 * List all threat feeds
 */
export async function listFeeds(): Promise<FeedListResponse> {
  return apiGet<FeedListResponse>('/api/v1/feeds');
}

/**
 * Get feed by ID
 */
export async function getFeed(feedId: string): Promise<ThreatFeed> {
  return apiGet<ThreatFeed>(`/api/v1/feeds/${feedId}`);
}

/**
 * Create a new feed
 */
export async function createFeed(request: CreateFeedRequest): Promise<ThreatFeed> {
  return apiPost<ThreatFeed>('/api/v1/feeds', request);
}

/**
 * Update a feed
 */
export async function updateFeed(
  feedId: string,
  request: UpdateFeedRequest
): Promise<ThreatFeed> {
  return apiPut<ThreatFeed>(`/api/v1/feeds/${feedId}`, request);
}

/**
 * Delete a feed
 */
export async function deleteFeed(feedId: string): Promise<void> {
  return apiDelete(`/api/v1/feeds/${feedId}`);
}

/**
 * Toggle feed active status
 */
export async function toggleFeed(feedId: string): Promise<ThreatFeed> {
  return apiPost<ThreatFeed>(`/api/v1/feeds/${feedId}/toggle`);
}

/**
 * Sync all feeds
 */
export async function syncAllFeeds(): Promise<any> {
  return apiPost('/api/v1/sync/all');
}

/**
 * Sync a specific feed
 */
export async function syncFeed(feedId: string): Promise<any> {
  return apiPost(`/api/v1/sync/${feedId}`);
}

/**
 * Get sync status
 */
export async function getSyncStatus(): Promise<any> {
  return apiGet('/api/v1/sync/status');
}
