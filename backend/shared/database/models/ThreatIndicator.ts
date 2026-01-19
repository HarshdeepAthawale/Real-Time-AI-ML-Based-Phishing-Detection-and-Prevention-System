import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';
import { Threat } from './Threat';

@Entity('threat_indicators')
export class ThreatIndicator {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  threat_id: string;

  @ManyToOne(() => Threat, (threat) => threat.indicators, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'threat_id' })
  threat: Threat;

  @Column({ type: 'varchar', length: 50 })
  indicator_type: string;

  @Column({ type: 'text' })
  indicator_value: string;

  @Column({ type: 'varchar', length: 50, nullable: true })
  source: string | null;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  first_seen_at: Date;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  last_seen_at: Date;

  @CreateDateColumn()
  created_at: Date;
}
