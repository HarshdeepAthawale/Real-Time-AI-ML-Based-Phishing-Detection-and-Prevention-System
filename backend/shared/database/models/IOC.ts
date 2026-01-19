import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  JoinColumn,
  OneToMany,
  Unique,
} from 'typeorm';
import { ThreatIntelligenceFeed } from './ThreatIntelligenceFeed';
import { IOCMatch } from './IOCMatch';

@Entity('iocs')
@Unique(['ioc_type', 'ioc_value_hash'])
export class IOC {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid', nullable: true })
  feed_id: string | null;

  @ManyToOne(() => ThreatIntelligenceFeed, (feed) => feed.iocs, { onDelete: 'SET NULL', nullable: true })
  @JoinColumn({ name: 'feed_id' })
  feed: ThreatIntelligenceFeed | null;

  @Column({ type: 'varchar', length: 50 })
  ioc_type: string;

  @Column({ type: 'text' })
  ioc_value: string;

  @Column({ type: 'varchar', length: 64, nullable: true })
  ioc_value_hash: string | null;

  @Column({ type: 'varchar', length: 100, nullable: true })
  threat_type: string | null;

  @Column({ type: 'varchar', length: 20, nullable: true })
  severity: string | null;

  @Column({ type: 'decimal', precision: 5, scale: 2, nullable: true })
  confidence: number | null;

  @Column({ type: 'timestamp', nullable: true })
  first_seen_at: Date | null;

  @Column({ type: 'timestamp', nullable: true })
  last_seen_at: Date | null;

  @Column({ type: 'int', default: 1 })
  source_reports: number;

  @Column({ type: 'jsonb', default: {} })
  metadata: Record<string, any>;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;

  @OneToMany(() => IOCMatch, (match) => match.ioc)
  matches: IOCMatch[];
}
