import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';
import { IOC } from './IOC';
import { Detection } from './Detection';

@Entity('ioc_matches')
export class IOCMatch {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  ioc_id: string;

  @ManyToOne(() => IOC, (ioc) => ioc.matches, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'ioc_id' })
  ioc: IOC;

  @Column({ type: 'uuid' })
  detection_id: string;

  @ManyToOne(() => Detection, (detection) => detection.ioc_matches, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'detection_id' })
  detection: Detection;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  matched_at: Date;
}
