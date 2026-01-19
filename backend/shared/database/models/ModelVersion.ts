import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';
import { MLModel } from './MLModel';

@Entity('model_versions')
export class ModelVersion {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  model_id: string;

  @ManyToOne(() => MLModel, (model) => model.versions, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'model_id' })
  model: MLModel;

  @Column({ type: 'varchar', length: 50 })
  version: string;

  @Column({ type: 'text', nullable: true })
  model_path_s3: string | null;

  @Column({ type: 'jsonb', default: {} })
  metrics: Record<string, any>;

  @Column({ type: 'uuid', nullable: true })
  training_job_id: string | null;

  @CreateDateColumn()
  created_at: Date;
}
