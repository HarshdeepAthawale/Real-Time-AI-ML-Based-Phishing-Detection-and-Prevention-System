import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
} from 'typeorm';

@Entity('training_jobs')
export class TrainingJob {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', length: 50 })
  model_type: string;

  @Column({ type: 'varchar', length: 20, default: 'pending' })
  status: string;

  @Column({ type: 'jsonb' })
  training_config: Record<string, any>;

  @Column({ type: 'text', nullable: true })
  dataset_path_s3: string | null;

  @Column({ type: 'int', nullable: true })
  dataset_size: number | null;

  @Column({ type: 'timestamp', nullable: true })
  started_at: Date | null;

  @Column({ type: 'timestamp', nullable: true })
  completed_at: Date | null;

  @Column({ type: 'jsonb', default: {} })
  metrics: Record<string, any>;

  @Column({ type: 'text', nullable: true })
  error_message: string | null;

  @Column({ type: 'text', nullable: true })
  logs_s3_path: string | null;

  @CreateDateColumn()
  created_at: Date;
}
