import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
  Unique,
} from 'typeorm';
import { MLModel } from './MLModel';

@Entity('model_performance')
@Unique(['model_id', 'date'])
export class ModelPerformance {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  model_id: string;

  @ManyToOne(() => MLModel, (model) => model.performance, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'model_id' })
  model: MLModel;

  @Column({ type: 'date' })
  date: Date;

  @Column({ type: 'int', default: 0 })
  total_predictions: number;

  @Column({ type: 'decimal', precision: 10, scale: 2, nullable: true })
  avg_inference_time_ms: number | null;

  @Column({ type: 'decimal', precision: 5, scale: 2, nullable: true })
  accuracy: number | null;

  @Column({ type: 'decimal', precision: 5, scale: 2, nullable: true })
  precision: number | null;

  @Column({ type: 'decimal', precision: 5, scale: 2, nullable: true })
  recall: number | null;

  @Column({ type: 'decimal', precision: 5, scale: 2, nullable: true })
  f1_score: number | null;

  @Column({ type: 'decimal', precision: 5, scale: 2, nullable: true })
  false_positive_rate: number | null;

  @CreateDateColumn()
  created_at: Date;
}
