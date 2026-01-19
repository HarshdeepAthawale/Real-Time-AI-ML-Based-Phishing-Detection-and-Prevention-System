import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';
import { Domain } from './Domain';

@Entity('urls')
export class URL {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  domain_id: string;

  @ManyToOne(() => Domain, (domain) => domain.urls, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'domain_id' })
  domain: Domain;

  @Column({ type: 'text' })
  full_url: string;

  @Column({ type: 'varchar', length: 64, unique: true })
  url_hash: string;

  @Column({ type: 'varchar', length: 10, nullable: true })
  scheme: string | null;

  @Column({ type: 'text', nullable: true })
  path: string | null;

  @Column({ type: 'jsonb', nullable: true })
  query_params: Record<string, any> | null;

  @Column({ type: 'text', nullable: true })
  fragment: string | null;

  @Column({ type: 'jsonb', nullable: true })
  redirect_chain: any[] | null;

  @Column({ type: 'int', default: 0 })
  redirect_count: number;

  @Column({ type: 'boolean', default: false })
  is_malicious: boolean;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  first_seen_at: Date;

  @Column({ type: 'timestamp', nullable: true })
  last_analyzed_at: Date | null;

  @Column({ type: 'jsonb', default: {} })
  analysis_metadata: Record<string, any>;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
