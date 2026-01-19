import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  OneToMany,
  Unique,
} from 'typeorm';
import { ModelVersion } from './ModelVersion';
import { ModelPerformance } from './ModelPerformance';

@Entity('ml_models')
@Unique(['model_type', 'version'])
export class MLModel {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', length: 50 })
  model_type: string;

  @Column({ type: 'varchar', length: 255 })
  name: string;

  @Column({ type: 'varchar', length: 50 })
  version: string;

  @Column({ type: 'text', nullable: true })
  description: string | null;

  @Column({ type: 'text', nullable: true })
  model_path_s3: string | null;

  @Column({ type: 'bigint', nullable: true })
  model_size_bytes: number | null;

  @Column({ type: 'varchar', length: 50, nullable: true })
  framework: string | null;

  @Column({ type: 'jsonb', nullable: true })
  input_schema: Record<string, any> | null;

  @Column({ type: 'jsonb', nullable: true })
  output_schema: Record<string, any> | null;

  @Column({ type: 'jsonb', default: {} })
  metrics: Record<string, any>;

  @Column({ type: 'jsonb', default: {} })
  training_config: Record<string, any>;

  @Column({ type: 'boolean', default: false })
  is_active: boolean;

  @Column({ type: 'timestamp', nullable: true })
  deployed_at: Date | null;

  @CreateDateColumn()
  created_at: Date;

  @OneToMany(() => ModelVersion, (version) => version.model)
  versions: ModelVersion[];

  @OneToMany(() => ModelPerformance, (perf) => perf.model)
  performance: ModelPerformance[];
}
