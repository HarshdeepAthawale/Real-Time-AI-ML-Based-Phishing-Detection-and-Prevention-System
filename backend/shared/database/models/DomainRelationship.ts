import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
  Unique,
} from 'typeorm';
import { Domain } from './Domain';

@Entity('domain_relationships')
@Unique(['source_domain_id', 'target_domain_id', 'relationship_type'])
export class DomainRelationship {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  source_domain_id: string;

  @ManyToOne(() => Domain, (domain) => domain.source_relationships, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'source_domain_id' })
  source_domain: Domain;

  @Column({ type: 'uuid' })
  target_domain_id: string;

  @ManyToOne(() => Domain, (domain) => domain.target_relationships, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'target_domain_id' })
  target_domain: Domain;

  @Column({ type: 'varchar', length: 50 })
  relationship_type: string;

  @Column({ type: 'decimal', precision: 5, scale: 2, default: 1.0 })
  strength: number;

  @Column({ type: 'jsonb', default: {} })
  metadata: Record<string, any>;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  first_seen_at: Date;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  last_seen_at: Date;

  @CreateDateColumn()
  created_at: Date;
}
