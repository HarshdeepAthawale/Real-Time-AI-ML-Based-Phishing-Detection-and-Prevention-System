import { IOCManagerService } from '../../../src/services/ioc-manager.service';
import { IOC } from '../../../src/models/ioc.model';
import { DataSource, Repository } from 'typeorm';
import { createMockDataSource } from '../../helpers/test-setup';

// Mock logger
jest.mock('../../../src/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('IOCManagerService', () => {
  let service: IOCManagerService;
  let mockDataSource: jest.Mocked<DataSource>;
  let mockRepository: jest.Mocked<Repository<any>>;

  beforeEach(() => {
    mockDataSource = createMockDataSource();
    mockRepository = {
      find: jest.fn(),
      findOne: jest.fn(),
      create: jest.fn(),
      save: jest.fn(),
      remove: jest.fn(),
      count: jest.fn(),
      createQueryBuilder: jest.fn(),
    } as any;
    mockDataSource.getRepository.mockReturnValue(mockRepository);

    service = new IOCManagerService(mockDataSource);
  });

  describe('createIOC', () => {
    it('should create new IOC', async () => {
      const newIOC: Omit<IOC, 'id' | 'createdAt' | 'updatedAt'> = {
        iocType: 'domain',
        iocValue: 'malicious.example.com',
        threatType: 'phishing',
        severity: 'high',
        confidence: 90,
        source: 'feed',
      };

      const savedEntity = {
        id: '123',
        ioc_type: 'domain',
        ioc_value: 'malicious.example.com',
        threat_type: 'phishing',
        severity: 'high',
        confidence: 90,
        feed_id: null,
        source_reports: 1,
        metadata: {},
        created_at: new Date(),
        updated_at: new Date(),
        first_seen_at: new Date(),
        last_seen_at: new Date(),
        ioc_value_hash: 'hash123',
      };

      mockRepository.findOne.mockResolvedValue(null);
      mockRepository.create.mockReturnValue(savedEntity as any);
      mockRepository.save.mockResolvedValue(savedEntity as any);

      const result = await service.createIOC(newIOC);

      expect(result).toBeDefined();
      expect(result.iocType).toBe('domain');
      expect(result.iocValue).toBe('malicious.example.com');
    });

    it('should update existing IOC on conflict', async () => {
      const existingEntity = {
        id: '123',
        ioc_type: 'domain',
        ioc_value: 'malicious.example.com',
        threat_type: 'phishing',
        severity: 'high',
        confidence: 90,
        feed_id: null,
        source_reports: 2,
        metadata: {},
        created_at: new Date(),
        updated_at: new Date(),
        first_seen_at: new Date(),
        last_seen_at: new Date(),
        ioc_value_hash: 'hash123',
      };

      mockRepository.findOne.mockResolvedValue(existingEntity as any);
      mockRepository.save.mockResolvedValue({
        ...existingEntity,
        source_reports: 3,
      } as any);

      const ioc: Omit<IOC, 'id' | 'createdAt' | 'updatedAt'> = {
        iocType: 'domain',
        iocValue: 'malicious.example.com',
        source: 'feed',
      };

      const result = await service.createIOC(ioc);

      expect(result.sourceReports).toBe(3);
    });
  });

  describe('findIOC', () => {
    it('should find IOC by type and value', async () => {
      const entity = {
        id: '123',
        ioc_type: 'domain',
        ioc_value: 'malicious.example.com',
        ioc_value_hash: 'hash123',
        threat_type: 'phishing',
        severity: 'high',
        confidence: 90,
        feed_id: null,
        source_reports: 1,
        metadata: {},
        created_at: new Date(),
        updated_at: new Date(),
        first_seen_at: new Date(),
        last_seen_at: new Date(),
      };

      mockRepository.findOne.mockResolvedValue(entity as any);

      const result = await service.findIOC('domain', 'malicious.example.com');

      expect(result).toBeDefined();
      expect(result?.iocValue).toBe('malicious.example.com');
    });

    it('should return null when IOC not found', async () => {
      mockRepository.findOne.mockResolvedValue(null);

      const result = await service.findIOC('domain', 'nonexistent.com');

      expect(result).toBeNull();
    });
  });

  describe('bulkCreateIOCs', () => {
    it('should bulk create IOCs', async () => {
      const iocs: Array<Omit<IOC, 'id' | 'createdAt' | 'updatedAt'>> = [
        {
          iocType: 'domain',
          iocValue: 'malicious1.com',
          source: 'feed',
        },
        {
          iocType: 'domain',
          iocValue: 'malicious2.com',
          source: 'feed',
        },
      ];

      mockRepository.findOne.mockResolvedValue(null);
      mockRepository.create.mockImplementation((entity: any) => entity);
      mockRepository.save.mockImplementation((entity: any) => Promise.resolve({
        ...entity,
        id: '123',
      }));

      const count = await service.bulkCreateIOCs(iocs);

      expect(count).toBe(2);
    });
  });

  describe('getStats', () => {
    it('should return IOC statistics', async () => {
      const mockQueryBuilder = {
        select: jest.fn().mockReturnThis(),
        addSelect: jest.fn().mockReturnThis(),
        where: jest.fn().mockReturnThis(),
        groupBy: jest.fn().mockReturnThis(),
        getRawMany: jest.fn(),
        getCount: jest.fn(),
      };

      mockRepository.count.mockResolvedValue(1000);
      mockRepository.createQueryBuilder.mockReturnValue(mockQueryBuilder as any);
      
      mockQueryBuilder.getRawMany
        .mockResolvedValueOnce([
          { type: 'domain', count: '500' },
          { type: 'ip', count: '300' },
        ])
        .mockResolvedValueOnce([
          { severity: 'high', count: '400' },
          { severity: 'medium', count: '300' },
        ])
        .mockResolvedValueOnce([
          { feedId: 'feed1', count: '600' },
          { feedId: 'feed2', count: '400' },
        ]);
      
      mockQueryBuilder.getCount.mockResolvedValue(50);

      const stats = await service.getStats();

      expect(stats.total).toBe(1000);
      expect(stats.byType.domain).toBe(500);
      expect(stats.recentCount).toBe(50);
    });
  });
});
