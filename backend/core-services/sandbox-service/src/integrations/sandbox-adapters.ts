import { BaseSandboxClient, SandboxResult, SandboxSubmitOptions } from './base-sandbox.client';
import { CuckooClient, CuckooAnalysisResult } from './cuckoo.client';
import { AnyRunClient, AnyRunAnalysisResult } from './anyrun.client';

function mapCuckooToSandboxResult(taskId: number, status: string, report?: CuckooAnalysisResult): SandboxResult {
  const sr: SandboxResult = {
    jobId: String(taskId),
    status: status as SandboxResult['status'],
    analysisId: String(taskId),
    results: report ? {
      network: report.network ? [{
        protocol: 'tcp',
        destination: (report.network.hosts || [])[0] || '',
        port: 0
      }] : undefined,
      processes: report.processes?.map(p => ({
        name: p.process_name,
        pid: p.process_id,
        commandLine: p.command_line,
        parentPid: p.parent_id
      })),
      score: report.score
    } : undefined,
    error: report ? undefined : 'Analysis not completed'
  };
  return Object.assign(sr, report || {}, { jobId: String(taskId), status, analysisId: String(taskId) });
}

function mapAnyRunToSandboxResult(taskId: string, result: AnyRunAnalysisResult): SandboxResult {
  return Object.assign({
    jobId: taskId,
    status: result.status,
    analysisId: taskId,
    results: {
      network: result.network?.connections?.map(c => ({
        protocol: c.protocol,
        destination: c.destination,
        port: c.port
      })),
      processes: result.processes?.map(p => ({
        name: p.name,
        pid: p.pid,
        commandLine: p.commandLine
      })),
      score: result.threatLevel
    }
  }, result, { jobId: taskId, analysisId: taskId });
}

/** No-op adapter when no sandbox provider is configured */
export class DisabledSandboxAdapter extends BaseSandboxClient {
  private fail(action: string): never {
    throw new Error(`Sandbox analysis is disabled. ${action} requires a configured provider (set ANYRUN_API_KEY or CUCKOO_SANDBOX_URL).`);
  }

  async submitFile(_fileBuffer: Buffer, _filename: string): Promise<string> {
    this.fail('File submission');
  }

  async submitURL(_url: string): Promise<string> {
    this.fail('URL submission');
  }

  async getStatus(_jobId: string): Promise<SandboxResult> {
    return { jobId: _jobId, status: 'failed', error: 'Sandbox provider not configured' };
  }

  async getResults(_analysisId: string): Promise<SandboxResult> {
    return { jobId: _analysisId, status: 'failed', error: 'Sandbox provider not configured' };
  }
}

/** Adapter: CuckooClient -> BaseSandboxClient */
export class CuckooSandboxAdapter extends BaseSandboxClient {
  constructor(private client: CuckooClient) {
    super();
  }

  async submitFile(fileBuffer: Buffer, filename: string, options?: SandboxSubmitOptions): Promise<string> {
    const r = await this.client.submitFile(fileBuffer, filename, options as any);
    return String(r.taskId);
  }

  async submitURL(url: string, options?: SandboxSubmitOptions): Promise<string> {
    const r = await this.client.submitURL(url, options as any);
    return String(r.taskId);
  }

  async getStatus(jobId: string): Promise<SandboxResult> {
    const status = await this.client.getTaskStatus(parseInt(jobId, 10));
    return mapCuckooToSandboxResult(parseInt(jobId, 10), status.status);
  }

  async getResults(jobId: string): Promise<SandboxResult> {
    const report = await this.client.getReport(parseInt(jobId, 10));
    return mapCuckooToSandboxResult(report.taskId, report.status, report);
  }
}

/** Adapter: AnyRunClient -> BaseSandboxClient */
export class AnyRunSandboxAdapter extends BaseSandboxClient {
  constructor(private client: AnyRunClient) {
    super();
  }

  async submitFile(fileBuffer: Buffer, filename: string, options?: SandboxSubmitOptions): Promise<string> {
    const r = await this.client.submitFile(fileBuffer, filename, options as any);
    return String(r.taskId);
  }

  async submitURL(url: string, options?: SandboxSubmitOptions): Promise<string> {
    const r = await this.client.submitURL(url, options as any);
    return String(r.taskId);
  }

  async getStatus(jobId: string): Promise<SandboxResult> {
    const result = await this.client.getAnalysis(jobId);
    return mapAnyRunToSandboxResult(jobId, result);
  }

  async getResults(jobId: string): Promise<SandboxResult> {
    const result = await this.client.getAnalysis(jobId);
    return mapAnyRunToSandboxResult(jobId, result);
  }
}
