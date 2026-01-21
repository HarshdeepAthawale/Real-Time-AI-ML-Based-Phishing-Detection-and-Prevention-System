// Type declarations for shared modules
declare module '../../shared/database' {
  import { DataSource } from 'typeorm';
  export function getPostgreSQL(): DataSource;
  export function connectPostgreSQL(): Promise<DataSource>;
  export function disconnectPostgreSQL(): Promise<void>;
}

declare module '../../shared/database/models' {
  export class ApiKey {
    id: string;
    key_prefix: string;
    key_hash: string;
    organization_id: string;
    revoked_at: Date | null;
    expires_at: Date | null;
    last_used_at: Date | null;
  }
}

declare module '../../../shared/database' {
  import { DataSource } from 'typeorm';
  export function getPostgreSQL(): DataSource;
  export function connectPostgreSQL(): Promise<DataSource>;
  export function disconnectPostgreSQL(): Promise<void>;
}

declare module '../../../shared/database/models' {
  export class ApiKey {
    id: string;
    key_prefix: string;
    key_hash: string;
    organization_id: string;
    revoked_at: Date | null;
    expires_at: Date | null;
    last_used_at: Date | null;
  }
}

declare module '../../../../shared/database' {
  import { DataSource } from 'typeorm';
  export function getPostgreSQL(): DataSource;
  export function connectPostgreSQL(): Promise<DataSource>;
  export function disconnectPostgreSQL(): Promise<void>;
}

declare module '../../../../shared/database/models' {
  export class ApiKey {
    id: string;
    key_prefix: string;
    key_hash: string;
    organization_id: string;
    revoked_at: Date | null;
    expires_at: Date | null;
    last_used_at: Date | null;
  }
  export class Organization {
    id: string;
    name: string;
    domain: string | null;
    plan: string;
    max_users: number;
    max_api_calls_per_day: number;
    created_at: Date;
    updated_at: Date;
    deleted_at: Date | null;
  }
  export class User {
    id: string;
    organization_id: string;
    email: string;
    password_hash: string;
    first_name: string | null;
    last_name: string | null;
    role: string;
    is_active: boolean;
    last_login_at: Date | null;
    created_at: Date;
    updated_at: Date;
    deleted_at: Date | null;
  }
}
