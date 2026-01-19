import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';
import { EmailMessage } from './EmailMessage';

@Entity('email_headers')
export class EmailHeader {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  email_message_id: string;

  @ManyToOne(() => EmailMessage, (email) => email.headers, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'email_message_id' })
  email_message: EmailMessage;

  @Column({ type: 'varchar', length: 255 })
  header_name: string;

  @Column({ type: 'text', nullable: true })
  header_value: string | null;

  @CreateDateColumn()
  created_at: Date;
}
