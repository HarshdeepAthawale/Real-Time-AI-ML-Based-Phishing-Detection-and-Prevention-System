import { SandboxResult, NetworkActivity, FileSystemActivity, ProcessActivity } from '../../src/integrations/base-sandbox.client';
import { FileAnalysis } from '../../src/services/file-analyzer.service';
import { BehavioralAnalysis } from '../../src/services/behavioral-analyzer.service';

export const mockFileBuffer = Buffer.from('test file content');

export const mockFileAnalysis: FileAnalysis = {
  filename: 'test.exe',
  fileType: 'exe',
  mimeType: 'application/x-msdownload',
  size: 1024,
  hash: {
    md5: '098f6bcd4621d373cade4e832627b4f6',
    sha1: 'a94a8fe5ccb19ba61c4c0873d391e987982fbbd3',
    sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
  },
  metadata: {},
  isExecutable: true,
  requiresSandbox: true,
};

export const mockNetworkActivity: NetworkActivity[] = [
  {
    protocol: 'tcp',
    destination: '192.168.1.100',
    port: 4444,
    method: 'POST',
    path: '/data',
    statusCode: 200,
  },
];

export const mockFileSystemActivity: FileSystemActivity[] = [
  {
    action: 'created',
    path: 'C:\\Windows\\System32\\malware.dll',
    fileType: 'dll',
  },
];

export const mockProcessActivity: ProcessActivity[] = [
  {
    name: 'cmd.exe',
    pid: 1234,
    commandLine: 'cmd.exe /c malicious_command',
    parentPid: 1,
  },
];

export const mockSandboxResult: SandboxResult = {
  jobId: 'test-job-123',
  status: 'completed',
  analysisId: 'test-analysis-123',
  results: {
    network: mockNetworkActivity,
    filesystem: mockFileSystemActivity,
    processes: mockProcessActivity,
    screenshots: ['screenshot1.png'],
    signatures: ['malware.signature'],
    score: 8.5,
  },
};

export const mockBehavioralAnalysis: BehavioralAnalysis = {
  isMalicious: true,
  threatScore: 85,
  indicators: ['c2_communication', 'data_exfiltration', 'suspicious_processes'],
  networkActivity: {
    suspiciousConnections: 1,
    c2Connections: 1,
    dataExfiltration: true,
  },
  fileSystemActivity: {
    suspiciousModifications: 1,
    systemFileAccess: true,
  },
  processActivity: {
    suspiciousProcesses: 1,
    processInjection: false,
  },
};

export const mockSandboxAnalysisRecord = {
  id: 'test-analysis-id',
  organization_id: 'test-org-id',
  analysis_type: 'file',
  target_file_hash: mockFileAnalysis.hash.sha256,
  sandbox_provider: 'anyrun',
  sandbox_job_id: 'test-job-123',
  status: 'pending',
  submitted_at: new Date(),
  started_at: null,
  completed_at: null,
  result_data: null,
  threat_id: null,
  created_at: new Date(),
  updated_at: new Date(),
};

export const mockThreatRecord = {
  id: 'test-threat-id',
  organization_id: 'test-org-id',
  threat_type: 'malware',
  severity: 'high',
  status: 'detected',
  confidence_score: 85,
  source: 'sandbox',
  source_value: mockFileAnalysis.hash.sha256,
  title: 'Sandbox Analysis: malware',
  description: 'Malicious behavior detected',
  metadata: {},
  detected_at: new Date(),
  created_at: new Date(),
  updated_at: new Date(),
};
