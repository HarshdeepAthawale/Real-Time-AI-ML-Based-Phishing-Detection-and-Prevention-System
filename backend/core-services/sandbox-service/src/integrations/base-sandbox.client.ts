export interface SandboxResult {
  jobId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  analysisId?: string;
  results?: {
    network?: NetworkActivity[];
    filesystem?: FileSystemActivity[];
    processes?: ProcessActivity[];
    registry?: RegistryActivity[];
    screenshots?: string[];
    signatures?: string[];
    score?: number;
  };
  error?: string;
}

export interface NetworkActivity {
  protocol: string;
  destination: string;
  port: number;
  method?: string;
  path?: string;
  statusCode?: number;
}

export interface FileSystemActivity {
  action: 'created' | 'modified' | 'deleted' | 'read';
  path: string;
  fileType?: string;
}

export interface ProcessActivity {
  name: string;
  pid: number;
  commandLine: string;
  parentPid?: number;
}

export interface RegistryActivity {
  action: 'created' | 'modified' | 'deleted';
  key: string;
  value?: string;
}

export interface SandboxSubmitOptions {
  timeout?: number;
  [key: string]: any;
}

export abstract class BaseSandboxClient {
  abstract submitFile(fileBuffer: Buffer, filename: string, options?: SandboxSubmitOptions): Promise<string>;
  abstract submitURL(url: string, options?: SandboxSubmitOptions): Promise<string>;
  abstract getStatus(jobId: string): Promise<SandboxResult>;
  abstract getResults(analysisId: string): Promise<SandboxResult>;
}
