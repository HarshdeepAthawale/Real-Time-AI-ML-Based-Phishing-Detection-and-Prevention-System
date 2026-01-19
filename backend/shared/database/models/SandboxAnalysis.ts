import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';
import { Organization } from './Organization';
import { Threat } from './Threat';

@Entity('sandbox_analyses')
export class SandboxAnalysis {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  organization_id: string;

  @ManyToOne(() => Organization, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'organization_id' })
  organization: Organization;

  @Column({ type: 'varchar', length: 50 })
  analysis_type: string;

  @Column({ type: 'text', nullable: true })
  target_url: string | null;

  @Column({ type: 'varchar', length: 64, nullable: true })
  target_file_hash: string | null;

  @Column({ type: 'varchar', length: 50, nullable: true })
  sandbox_provider: string | null;

  @Column({ type: 'varchar', length: 255, nullable: true })
  sandbox_job_id: string | null;

  @Column({ type: 'varchar', length: 20, default: 'pending' })
  status: string;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  submitted_at: Date;

  @Column({ type: 'timestamp', nullable: true })
  started_at: Date | null;

  @Column({ type: 'timestamp', nullable: true })
  completed_at: Date | null;

  @Column({ type: 'jsonb', nullable: true })
  result_data: Record<string, any> | null;

  @Column({ type: 'uuid', nullable: true })
  threat_id: string | null;

  @ManyToOne(() => Threat, { onDelete: 'SET NULL', nullable: true })
  @JoinColumn({ name: 'threat_id' })
  threat: Threat | null;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
