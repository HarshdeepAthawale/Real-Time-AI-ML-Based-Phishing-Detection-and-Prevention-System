import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  DeleteDateColumn,
  ManyToOne,
  JoinColumn,
  OneToMany,
} from 'typeorm';
import { Organization } from './Organization';
import { ApiKey } from './ApiKey';
import { DetectionFeedback } from './DetectionFeedback';

@Entity('users')
export class User {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  organization_id: string;

  @ManyToOne(() => Organization, (org) => org.users, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'organization_id' })
  organization: Organization;

  @Column({ type: 'varchar', length: 255, unique: true })
  email: string;

  @Column({ type: 'varchar', length: 255 })
  password_hash: string;

  @Column({ type: 'varchar', length: 100, nullable: true })
  first_name: string | null;

  @Column({ type: 'varchar', length: 100, nullable: true })
  last_name: string | null;

  @Column({ type: 'varchar', length: 50, default: 'user' })
  role: string;

  @Column({ type: 'boolean', default: true })
  is_active: boolean;

  @Column({ type: 'timestamp', nullable: true })
  last_login_at: Date | null;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;

  @DeleteDateColumn({ nullable: true })
  deleted_at: Date | null;

  @OneToMany(() => ApiKey, (apiKey) => apiKey.user)
  api_keys: ApiKey[];

  @OneToMany(() => DetectionFeedback, (feedback) => feedback.user)
  feedback: DetectionFeedback[];
}
