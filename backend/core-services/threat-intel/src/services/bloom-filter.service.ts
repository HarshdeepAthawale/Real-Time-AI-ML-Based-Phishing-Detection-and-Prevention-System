import { createHash } from 'crypto';
import { logger } from '../utils/logger';

/**
 * Bloom Filter implementation for fast IOC lookups
 * 
 * A Bloom filter is a space-efficient probabilistic data structure
 * that is used to test whether an element is a member of a set.
 * 
 * False positive matches are possible, but false negatives are not.
 * If the filter returns false, the element is definitely not in the set.
 * If it returns true, the element *might* be in the set (needs verification).
 */
export class BloomFilterService {
  private bitArray: Uint8Array;
  private size: number;
  private numHashes: number;
  private itemCount: number = 0;

  constructor(
    expectedItems: number = 1000000,
    falsePositiveRate: number = 0.01
  ) {
    // Calculate optimal size and number of hash functions
    this.size = this.calculateOptimalSize(expectedItems, falsePositiveRate);
    this.numHashes = this.calculateOptimalHashCount(this.size, expectedItems);
    
    // Initialize bit array (using bytes for efficiency)
    const byteSize = Math.ceil(this.size / 8);
    this.bitArray = new Uint8Array(byteSize);
    
    logger.info(
      `Bloom filter initialized: ${this.size} bits, ${this.numHashes} hash functions, ` +
      `${(byteSize / 1024 / 1024).toFixed(2)} MB memory`
    );
  }

  /**
   * Calculate optimal Bloom filter size
   * m = -(n * ln(p)) / (ln(2)^2)
   * where n = expected items, p = false positive rate
   */
  private calculateOptimalSize(n: number, p: number): number {
    const size = Math.ceil(-(n * Math.log(p)) / (Math.log(2) ** 2));
    return size;
  }

  /**
   * Calculate optimal number of hash functions
   * k = (m/n) * ln(2)
   * where m = size, n = expected items
   */
  private calculateOptimalHashCount(m: number, n: number): number {
    const k = Math.ceil((m / n) * Math.log(2));
    return Math.max(1, Math.min(k, 10)); // Cap at 10 hashes for performance
  }

  /**
   * Generate hash values for an item
   */
  private getHashes(item: string): number[] {
    const hashes: number[] = [];
    
    // Use double hashing technique: h1 and h2 are independent hash functions
    const h1 = this.hash(item, 'md5');
    const h2 = this.hash(item, 'sha1');
    
    for (let i = 0; i < this.numHashes; i++) {
      // Combined hash: h(i) = (h1 + i * h2) % m
      const combined = (h1 + i * h2) % this.size;
      hashes.push(Math.abs(combined));
    }
    
    return hashes;
  }

  /**
   * Hash function using crypto
   */
  private hash(item: string, algorithm: 'md5' | 'sha1'): number {
    const hash = createHash(algorithm).update(item).digest();
    
    // Convert first 4 bytes to number
    return (hash[0] << 24) | (hash[1] << 16) | (hash[2] << 8) | hash[3];
  }

  /**
   * Set a bit in the bit array
   */
  private setBit(index: number): void {
    const byteIndex = Math.floor(index / 8);
    const bitIndex = index % 8;
    this.bitArray[byteIndex] |= (1 << bitIndex);
  }

  /**
   * Check if a bit is set
   */
  private isBitSet(index: number): boolean {
    const byteIndex = Math.floor(index / 8);
    const bitIndex = index % 8;
    return (this.bitArray[byteIndex] & (1 << bitIndex)) !== 0;
  }

  /**
   * Add an item to the Bloom filter
   */
  add(item: string): void {
    const hashes = this.getHashes(item.toLowerCase());
    
    for (const hash of hashes) {
      this.setBit(hash);
    }
    
    this.itemCount++;
  }

  /**
   * Add multiple items to the Bloom filter
   */
  addBatch(items: string[]): void {
    for (const item of items) {
      this.add(item);
    }
  }

  /**
   * Check if an item might be in the set
   * Returns true if item *might* be in the set (need to verify)
   * Returns false if item is *definitely not* in the set
   */
  mightContain(item: string): boolean {
    const hashes = this.getHashes(item.toLowerCase());
    
    for (const hash of hashes) {
      if (!this.isBitSet(hash)) {
        return false; // Definitely not in set
      }
    }
    
    return true; // Might be in set (needs verification)
  }

  /**
   * Clear the Bloom filter
   */
  clear(): void {
    this.bitArray.fill(0);
    this.itemCount = 0;
    logger.info('Bloom filter cleared');
  }

  /**
   * Get current false positive probability
   * p = (1 - e^(-kn/m))^k
   * where k = num hashes, n = items added, m = size
   */
  getCurrentFalsePositiveRate(): number {
    if (this.itemCount === 0) return 0;
    
    const exponent = -(this.numHashes * this.itemCount) / this.size;
    const probability = Math.pow(1 - Math.exp(exponent), this.numHashes);
    
    return probability;
  }

  /**
   * Get fill rate (percentage of bits set to 1)
   */
  getFillRate(): number {
    let setBits = 0;
    
    for (let i = 0; i < this.bitArray.length; i++) {
      let byte = this.bitArray[i];
      while (byte) {
        setBits += byte & 1;
        byte >>= 1;
      }
    }
    
    return (setBits / this.size) * 100;
  }

  /**
   * Get statistics
   */
  getStats(): {
    size: number;
    numHashes: number;
    itemCount: number;
    fillRate: number;
    falsePositiveRate: number;
    memorySizeMB: number;
  } {
    return {
      size: this.size,
      numHashes: this.numHashes,
      itemCount: this.itemCount,
      fillRate: this.getFillRate(),
      falsePositiveRate: this.getCurrentFalsePositiveRate(),
      memorySizeMB: (this.bitArray.length / 1024 / 1024)
    };
  }

  /**
   * Serialize Bloom filter to JSON
   */
  toJSON(): {
    size: number;
    numHashes: number;
    itemCount: number;
    bitArray: number[];
  } {
    return {
      size: this.size,
      numHashes: this.numHashes,
      itemCount: this.itemCount,
      bitArray: Array.from(this.bitArray)
    };
  }

  /**
   * Deserialize Bloom filter from JSON
   */
  static fromJSON(data: {
    size: number;
    numHashes: number;
    itemCount: number;
    bitArray: number[];
  }): BloomFilterService {
    const filter = new BloomFilterService(1, 0.01); // Create with dummy values
    filter.size = data.size;
    filter.numHashes = data.numHashes;
    filter.itemCount = data.itemCount;
    filter.bitArray = new Uint8Array(data.bitArray);
    
    return filter;
  }

  /**
   * Export to Redis-compatible format (base64 string)
   */
  toRedisString(): string {
    const json = this.toJSON();
    return Buffer.from(JSON.stringify(json)).toString('base64');
  }

  /**
   * Import from Redis format
   */
  static fromRedisString(data: string): BloomFilterService {
    const json = JSON.parse(Buffer.from(data, 'base64').toString('utf-8'));
    return BloomFilterService.fromJSON(json);
  }

  /**
   * Estimate memory usage in bytes
   */
  getMemoryUsage(): number {
    return this.bitArray.length;
  }

  /**
   * Check if filter needs rebuilding (too full)
   */
  needsRebuilding(threshold: number = 0.7): boolean {
    const fillRate = this.getFillRate() / 100;
    return fillRate > threshold;
  }
}
