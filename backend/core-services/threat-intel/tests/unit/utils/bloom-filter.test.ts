import {
  createBloomFilter,
  serializeBloomFilter,
  loadBloomFilter,
  estimateBloomFilterSize,
  bulkAddToBloomFilter,
} from '../../../src/utils/bloom-filter';
import { BloomFilter } from 'bloom-filters';

describe('Bloom Filter Utilities', () => {
  describe('createBloomFilter', () => {
    it('should create a bloom filter with specified configuration', () => {
      const filter = createBloomFilter({
        size: 100,
        falsePositiveRate: 0.01,
      });
      
      expect(filter).toBeInstanceOf(BloomFilter);
    });
  });

  describe('serializeBloomFilter', () => {
    it('should serialize bloom filter to JSON', () => {
      const filter = createBloomFilter({
        size: 100,
        falsePositiveRate: 0.01,
      });
      
      const serialized = serializeBloomFilter(filter);
      expect(serialized).toBeDefined();
      expect(serialized).toHaveProperty('type');
      expect(serialized).toHaveProperty('_size');
      expect(serialized).toHaveProperty('_filter');
    });
  });

  describe('loadBloomFilter', () => {
    it('should load bloom filter from serialized data', () => {
      const original = createBloomFilter({
        size: 100,
        falsePositiveRate: 0.01,
      });
      
      original.add('test');
      const serialized = serializeBloomFilter(original);
      const loaded = loadBloomFilter(serialized);
      
      expect(loaded.has('test')).toBe(true);
      expect(loaded.has('not-present')).toBe(false);
    });
  });

  describe('estimateBloomFilterSize', () => {
    it('should estimate required size correctly', () => {
      const size = estimateBloomFilterSize(1000, 0.01);
      expect(size).toBeGreaterThan(0);
    });
  });

  describe('bulkAddToBloomFilter', () => {
    it('should add multiple items to bloom filter', () => {
      const filter = createBloomFilter({
        size: 1000,
        falsePositiveRate: 0.01,
      });
      
      const items = ['item1', 'item2', 'item3'];
      bulkAddToBloomFilter(filter, items);
      
      for (const item of items) {
        expect(filter.has(item)).toBe(true);
      }
    });
  });
});
