import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
  OneToMany,
} from 'typeorm';
import { Organization } from './Organization';
import { Threat } from './Threat';
import { EmailHeader } from './EmailHeader';

@Entity('email_messages')
export class EmailMessage {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  organization_id: string;

  @ManyToOne(() => Organization, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'organization_id' })
  organization: Organization;

  @Column({ type: 'varchar', length: 500, unique: true, nullable: true })
  message_id: string | null;

  @Column({ type: 'varchar', length: 255, nullable: true })
  from_email: string | null;

  @Column({ type: 'text', array: true, nullable: true })
  to_emails: string[] | null;

  @Column({ type: 'text', nullable: true })
  subject: string | null;

  @Column({ type: 'timestamp', nullable: true })
  received_at: Date | null;

  @Column({ type: 'timestamp', nullable: true })
  analyzed_at: Date | null;

  @Column({ type: 'uuid', nullable: true })
  threat_id: string | null;

  @ManyToOne(() => Threat, { onDelete: 'SET NULL', nullable: true })
  @JoinColumn({ name: 'threat_id' })
  threat: Threat | null;

  @CreateDateColumn()
  created_at: Date;

  @OneToMany(() => EmailHeader, (header) => header.email_message)
  headers: EmailHeader[];
}
