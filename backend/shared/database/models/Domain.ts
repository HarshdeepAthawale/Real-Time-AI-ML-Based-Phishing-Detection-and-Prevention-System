import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  OneToMany,
} from 'typeorm';
import { URL } from './URL';
import { DomainRelationship } from './DomainRelationship';

@Entity('domains')
export class Domain {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', length: 255, unique: true })
  domain: string;

  @Column({ type: 'varchar', length: 50, nullable: true })
  tld: string | null;

  @Column({ type: 'varchar', length: 255, nullable: true })
  subdomain: string | null;

  @Column({ type: 'varchar', length: 255, nullable: true })
  registered_domain: string | null;

  @Column({ type: 'decimal', precision: 5, scale: 2, default: 50 })
  reputation_score: number;

  @Column({ type: 'boolean', default: false })
  is_malicious: boolean;

  @Column({ type: 'boolean', default: false })
  is_suspicious: boolean;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  first_seen_at: Date;

  @Column({ type: 'timestamp', nullable: true })
  last_analyzed_at: Date | null;

  @Column({ type: 'jsonb', nullable: true })
  whois_data: Record<string, any> | null;

  @Column({ type: 'jsonb', nullable: true })
  dns_records: Record<string, any> | null;

  @Column({ type: 'jsonb', nullable: true })
  ssl_certificate_data: Record<string, any> | null;

  @Column({ type: 'jsonb', default: {} })
  analysis_metadata: Record<string, any>;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;

  @OneToMany(() => URL, (url) => url.domain)
  urls: URL[];

  @OneToMany(() => DomainRelationship, (rel) => rel.source_domain)
  source_relationships: DomainRelationship[];

  @OneToMany(() => DomainRelationship, (rel) => rel.target_domain)
  target_relationships: DomainRelationship[];
}
