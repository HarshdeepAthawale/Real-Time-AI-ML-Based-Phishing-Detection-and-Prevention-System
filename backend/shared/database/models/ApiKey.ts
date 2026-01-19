import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';
import { Organization } from './Organization';
import { User } from './User';

@Entity('api_keys')
export class ApiKey {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  organization_id: string;

  @ManyToOne(() => Organization, (org) => org.api_keys, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'organization_id' })
  organization: Organization;

  @Column({ type: 'uuid', nullable: true })
  user_id: string | null;

  @ManyToOne(() => User, (user) => user.api_keys, { onDelete: 'SET NULL', nullable: true })
  @JoinColumn({ name: 'user_id' })
  user: User | null;

  @Column({ type: 'varchar', length: 255, unique: true })
  key_hash: string;

  @Column({ type: 'varchar', length: 20 })
  key_prefix: string;

  @Column({ type: 'varchar', length: 255 })
  name: string;

  @Column({ type: 'jsonb', default: {} })
  permissions: Record<string, any>;

  @Column({ type: 'int', default: 100 })
  rate_limit_per_minute: number;

  @Column({ type: 'timestamp', nullable: true })
  last_used_at: Date | null;

  @Column({ type: 'timestamp', nullable: true })
  expires_at: Date | null;

  @CreateDateColumn()
  created_at: Date;

  @Column({ type: 'timestamp', nullable: true })
  revoked_at: Date | null;
}
