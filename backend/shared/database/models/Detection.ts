import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
  OneToMany,
} from 'typeorm';
import { Threat } from './Threat';
import { Organization } from './Organization';
import { DetectionFeedback } from './DetectionFeedback';
import { IOCMatch } from './IOCMatch';

@Entity('detections')
export class Detection {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid', nullable: true })
  threat_id: string | null;

  @ManyToOne(() => Threat, (threat) => threat.detections, { onDelete: 'SET NULL', nullable: true })
  @JoinColumn({ name: 'threat_id' })
  threat: Threat | null;

  @Column({ type: 'uuid' })
  organization_id: string;

  @ManyToOne(() => Organization, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'organization_id' })
  organization: Organization;

  @Column({ type: 'varchar', length: 50 })
  detection_type: string;

  @Column({ type: 'varchar', length: 50, nullable: true })
  model_version: string | null;

  @Column({ type: 'jsonb' })
  input_data: Record<string, any>;

  @Column({ type: 'jsonb' })
  analysis_result: Record<string, any>;

  @Column({ type: 'decimal', precision: 5, scale: 2 })
  confidence_score: number;

  @Column({ type: 'int', nullable: true })
  processing_time_ms: number | null;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  detected_at: Date;

  @CreateDateColumn()
  created_at: Date;

  @OneToMany(() => DetectionFeedback, (feedback) => feedback.detection)
  feedback: DetectionFeedback[];

  @OneToMany(() => IOCMatch, (match) => match.detection)
  ioc_matches: IOCMatch[];
}
