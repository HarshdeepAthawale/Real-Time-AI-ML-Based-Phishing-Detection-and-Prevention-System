import { IOCMatcherService } from '../../../src/services/ioc-matcher.service';
import { IOCManagerService } from '../../../src/services/ioc-manager.service';
import { IOC } from '../../../src/models/ioc.model';
import { createMockRedis } from '../../helpers/test-setup';

// Mock logger
jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('IOCMatcherService', () => {
  let service: IOCMatcherService;
  let mockRedis: any;
  let mockIocManager: jest.Mocked<IOCManagerService>;

  beforeEach(() => {
    mockRedis = createMockRedis();
    mockIocManager = {
      findIOC: jest.fn(),
      searchIOCs: jest.fn(),
    } as any;

    service = new IOCMatcherService(mockRedis, mockIocManager);
  });

  describe('matchIOC', () => {
    it('should return null when bloom filter indicates IOC not present', async () => {
      // Mock bloom filter to return false (not in filter)
      const mockFilter = {
        has: jest.fn().mockReturnValue(false),
      };
      
      (service as any).bloomFilters.set('domain', mockFilter);

      const result = await service.matchIOC('domain', 'nonexistent.com');

      expect(result).toBeNull();
      expect(mockIocManager.findIOC).not.toHaveBeenCalled();
    });

    it('should check database when bloom filter indicates IOC might be present', async () => {
      const mockIOC: IOC = {
        id: '123',
        iocType: 'domain',
        iocValue: 'malicious.com',
        source: 'feed',
      };

      const mockFilter = {
        has: jest.fn().mockReturnValue(true), // Might be in database
      };
      
      (service as any).bloomFilters.set('domain', mockFilter);
      mockIocManager.findIOC.mockResolvedValue(mockIOC);

      const result = await service.matchIOC('domain', 'malicious.com');

      expect(result).toEqual(mockIOC);
      expect(mockIocManager.findIOC).toHaveBeenCalledWith('domain', 'malicious.com');
    });
  });

  describe('addToBloomFilter', () => {
    it('should add IOC to bloom filter and persist to Redis', async () => {
      const mockFilter = {
        add: jest.fn(),
        saveAsJSON: jest.fn().mockReturnValue({ type: 'BloomFilter' }),
      };
      
      (service as any).bloomFilters.set('domain', mockFilter);
      mockRedis.setex.mockResolvedValue('OK');

      await service.addToBloomFilter('domain', 'malicious.com');

      expect(mockFilter.add).toHaveBeenCalled();
      expect(mockRedis.setex).toHaveBeenCalled();
    });
  });

  describe('bulkAddToBloomFilter', () => {
    it('should add multiple IOCs to bloom filter', async () => {
      const mockFilter = {
        add: jest.fn(),
        saveAsJSON: jest.fn().mockReturnValue({ type: 'BloomFilter' }),
      };
      
      (service as any).bloomFilters.set('domain', mockFilter);
      mockRedis.setex.mockResolvedValue('OK');

      const values = ['malicious1.com', 'malicious2.com', 'malicious3.com'];

      await service.bulkAddToBloomFilter('domain', values);

      expect(mockFilter.add).toHaveBeenCalledTimes(3);
      expect(mockRedis.setex).toHaveBeenCalled();
    });
  });

  describe('rebuildBloomFilter', () => {
    it('should rebuild bloom filter from database', async () => {
      const mockIOCs: IOC[] = [
        {
          id: '1',
          iocType: 'domain',
          iocValue: 'malicious1.com',
          source: 'feed',
        },
        {
          id: '2',
          iocType: 'domain',
          iocValue: 'malicious2.com',
          source: 'feed',
        },
      ];

      mockIocManager.searchIOCs.mockResolvedValue({
        iocs: mockIOCs,
        total: 2,
      });

      mockRedis.setex.mockResolvedValue('OK');

      await service.rebuildBloomFilter('domain');

      expect(mockIocManager.searchIOCs).toHaveBeenCalled();
      const filter = (service as any).bloomFilters.get('domain');
      expect(filter).toBeDefined();
    });
  });
});
