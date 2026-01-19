import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  DeleteDateColumn,
  OneToMany,
} from 'typeorm';
import { User } from './User';
import { ApiKey } from './ApiKey';
import { Threat } from './Threat';

@Entity('organizations')
export class Organization {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', length: 255 })
  name: string;

  @Column({ type: 'varchar', length: 255, unique: true, nullable: true })
  domain: string | null;

  @Column({ type: 'varchar', length: 50, default: 'free' })
  plan: string;

  @Column({ type: 'int', default: 10 })
  max_users: number;

  @Column({ type: 'int', default: 10000 })
  max_api_calls_per_day: number;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;

  @DeleteDateColumn({ nullable: true })
  deleted_at: Date | null;

  @OneToMany(() => User, (user) => user.organization)
  users: User[];

  @OneToMany(() => ApiKey, (apiKey) => apiKey.organization)
  api_keys: ApiKey[];

  @OneToMany(() => Threat, (threat) => threat.organization)
  threats: Threat[];
}
