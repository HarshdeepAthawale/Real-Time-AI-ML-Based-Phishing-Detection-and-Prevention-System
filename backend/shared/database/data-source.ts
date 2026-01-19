import { DataSource } from 'typeorm';
import { config } from '../config';
import * as entities from './models';

export const AppDataSource = new DataSource({
  type: 'postgres',
  url: config.database.url,
  entities: Object.values(entities),
  migrations: [__dirname + '/migrations/**/*.ts'],
  synchronize: false,
  logging: config.app.nodeEnv === 'development',
});
