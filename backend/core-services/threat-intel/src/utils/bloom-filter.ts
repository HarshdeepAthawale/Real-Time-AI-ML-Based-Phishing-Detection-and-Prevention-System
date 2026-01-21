import { BloomFilter } from 'bloom-filters';
import { logger } from './logger';

/**
 * Bloom filter utilities for fast IOC negative lookups
 */

export interface BloomFilterConfig {
  size: number;
  falsePositiveRate: number;
}

export interface SerializedBloomFilter {
  type: string;
  size: number;
  hashes: number;
  seed: number;
  buckets: string[];
}

/**
 * Create a new bloom filter with specified configuration
 */
export function createBloomFilter(config: BloomFilterConfig): BloomFilter {
  return BloomFilter.create(config.size, config.falsePositiveRate);
}

/**
 * Load bloom filter from JSON serialized data
 */
export function loadBloomFilter(serialized: SerializedBloomFilter): BloomFilter {
  try {
    return BloomFilter.fromJSON(serialized);
  } catch (error) {
    logger.error('Failed to load bloom filter from JSON', error);
    throw new Error('Invalid bloom filter data');
  }
}

/**
 * Serialize bloom filter to JSON
 */
export function serializeBloomFilter(filter: BloomFilter): SerializedBloomFilter {
  return filter.saveAsJSON() as SerializedBloomFilter;
}

/**
 * Estimate required size for bloom filter based on expected number of items
 */
export function estimateBloomFilterSize(
  expectedItems: number,
  falsePositiveRate: number
): number {
  // m = - (n * ln(p)) / (ln(2)^2)
  // where m = bits, n = items, p = false positive rate
  const ln2 = Math.log(2);
  const m = -((expectedItems * Math.log(falsePositiveRate)) / (ln2 * ln2));
  return Math.ceil(m / 8); // Convert to bytes, round up
}

/**
 * Add multiple items to bloom filter in batch
 */
export function bulkAddToBloomFilter(
  filter: BloomFilter,
  items: string[]
): void {
  for (const item of items) {
    filter.add(item);
  }
}

/**
 * Check if bloom filter needs to be rebuilt based on current size
 */
export function shouldRebuildBloomFilter(
  currentFilter: BloomFilter,
  newItemsCount: number,
  maxItems: number
): boolean {
  // If we're approaching the capacity, suggest rebuild
  // This is a simple heuristic - actual usage tracking would be better
  return newItemsCount > maxItems * 0.8;
}
