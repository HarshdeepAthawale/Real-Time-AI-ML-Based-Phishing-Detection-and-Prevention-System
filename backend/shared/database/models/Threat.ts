import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  JoinColumn,
  OneToMany,
} from 'typeorm';
import { Organization } from './Organization';
import { Detection } from './Detection';
import { ThreatIndicator } from './ThreatIndicator';

@Entity('threats')
export class Threat {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  organization_id: string;

  @ManyToOne(() => Organization, (org) => org.threats, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'organization_id' })
  organization: Organization;

  @Column({ type: 'varchar', length: 50 })
  threat_type: string;

  @Column({ type: 'varchar', length: 20 })
  severity: string;

  @Column({ type: 'varchar', length: 20, default: 'detected' })
  status: string;

  @Column({ type: 'decimal', precision: 5, scale: 2 })
  confidence_score: number;

  @Column({ type: 'varchar', length: 50, nullable: true })
  source: string | null;

  @Column({ type: 'text', nullable: true })
  source_value: string | null;

  @Column({ type: 'varchar', length: 500, nullable: true })
  title: string | null;

  @Column({ type: 'text', nullable: true })
  description: string | null;

  @Column({ type: 'jsonb', default: {} })
  metadata: Record<string, any>;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  detected_at: Date;

  @Column({ type: 'timestamp', nullable: true })
  resolved_at: Date | null;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;

  @OneToMany(() => Detection, (detection) => detection.threat)
  detections: Detection[];

  @OneToMany(() => ThreatIndicator, (indicator) => indicator.threat)
  indicators: ThreatIndicator[];
}
