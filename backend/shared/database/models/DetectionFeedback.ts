import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';
import { Detection } from './Detection';
import { User } from './User';

@Entity('detection_feedback')
export class DetectionFeedback {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  detection_id: string;

  @ManyToOne(() => Detection, (detection) => detection.feedback, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'detection_id' })
  detection: Detection;

  @Column({ type: 'uuid', nullable: true })
  user_id: string | null;

  @ManyToOne(() => User, (user) => user.feedback, { onDelete: 'SET NULL', nullable: true })
  @JoinColumn({ name: 'user_id' })
  user: User | null;

  @Column({ type: 'varchar', length: 20 })
  feedback_type: string;

  @Column({ type: 'text', nullable: true })
  comment: string | null;

  @CreateDateColumn()
  created_at: Date;
}
