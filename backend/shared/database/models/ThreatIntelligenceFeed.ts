import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  OneToMany,
} from 'typeorm';
import { IOC } from './IOC';

@Entity('threat_intelligence_feeds')
export class ThreatIntelligenceFeed {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', length: 255, unique: true })
  name: string;

  @Column({ type: 'varchar', length: 50 })
  feed_type: string;

  @Column({ type: 'text', nullable: true })
  api_endpoint: string | null;

  @Column({ type: 'text', nullable: true })
  api_key_encrypted: string | null;

  @Column({ type: 'int', default: 60 })
  sync_interval_minutes: number;

  @Column({ type: 'timestamp', nullable: true })
  last_sync_at: Date | null;

  @Column({ type: 'varchar', length: 20, nullable: true })
  last_sync_status: string | null;

  @Column({ type: 'text', nullable: true })
  last_sync_error: string | null;

  @Column({ type: 'boolean', default: true })
  is_active: boolean;

  @Column({ type: 'int', default: 0 })
  iocs_imported: number;

  @Column({ type: 'decimal', precision: 5, scale: 2, default: 50 })
  reliability_score: number;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;

  @OneToMany(() => IOC, (ioc) => ioc.feed)
  iocs: IOC[];
}
