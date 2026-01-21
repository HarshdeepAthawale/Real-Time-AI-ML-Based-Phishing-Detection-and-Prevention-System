import { SandboxResult } from '../integrations/base-sandbox.client';

export interface BehavioralAnalysis {
  isMalicious: boolean;
  threatScore: number;
  indicators: string[];
  networkActivity: {
    suspiciousConnections: number;
    c2Connections: number;
    dataExfiltration: boolean;
  };
  fileSystemActivity: {
    suspiciousModifications: number;
    systemFileAccess: boolean;
  };
  processActivity: {
    suspiciousProcesses: number;
    processInjection: boolean;
  };
}

export class BehavioralAnalyzerService {
  // Known C2 indicators (in production, this would query IOC database)
  private knownC2Domains: Set<string> = new Set();
  
  analyze(sandboxResult: SandboxResult): BehavioralAnalysis {
    const results = sandboxResult.results || {};
    const indicators: string[] = [];
    let threatScore = 0;
    
    // Analyze network activity
    const networkAnalysis = this.analyzeNetworkActivity(results.network || []);
    if (networkAnalysis.suspiciousConnections > 0) {
      indicators.push('suspicious_network_activity');
      threatScore += 20;
    }
    if (networkAnalysis.c2Connections > 0) {
      indicators.push('c2_communication');
      threatScore += 40;
    }
    if (networkAnalysis.dataExfiltration) {
      indicators.push('data_exfiltration');
      threatScore += 30;
    }
    
    // Analyze file system activity
    const fsAnalysis = this.analyzeFileSystemActivity(results.filesystem || []);
    if (fsAnalysis.suspiciousModifications > 0) {
      indicators.push('suspicious_file_modifications');
      threatScore += 15;
    }
    if (fsAnalysis.systemFileAccess) {
      indicators.push('system_file_access');
      threatScore += 25;
    }
    
    // Analyze process activity
    const processAnalysis = this.analyzeProcessActivity(results.processes || []);
    if (processAnalysis.suspiciousProcesses > 0) {
      indicators.push('suspicious_processes');
      threatScore += 20;
    }
    if (processAnalysis.processInjection) {
      indicators.push('process_injection');
      threatScore += 35;
    }
    
    // Check sandbox score
    if (results.score && results.score > 7) {
      indicators.push('high_sandbox_score');
      threatScore += 30;
    }
    
    // Check signatures
    if (results.signatures && results.signatures.length > 0) {
      indicators.push('malware_signatures_detected');
      threatScore += 25;
    }
    
    const isMalicious = threatScore >= 50;
    
    return {
      isMalicious,
      threatScore: Math.min(100, threatScore),
      indicators,
      networkActivity: networkAnalysis,
      fileSystemActivity: fsAnalysis,
      processActivity: processAnalysis
    };
  }
  
  private analyzeNetworkActivity(network: any[]): {
    suspiciousConnections: number;
    c2Connections: number;
    dataExfiltration: boolean;
  } {
    // Suspicious ports (commonly used by malware)
    const suspiciousPorts = [4444, 5555, 6666, 8080, 8443, 4443, 9999, 1337];
    const suspiciousConnections = network.filter(conn => 
      suspiciousPorts.includes(conn.port)
    ).length;
    
    // C2 indicators (known malicious domains/IPs)
    const c2Connections = network.filter(conn =>
      this.isKnownC2Domain(conn.destination)
    ).length;
    
    // Data exfiltration indicators (large outbound transfers, POST requests)
    const dataExfiltration = network.some(conn =>
      (conn.method === 'POST' && conn.statusCode === 200) ||
      (conn.method === 'PUT' && conn.statusCode === 200)
    );
    
    return {
      suspiciousConnections,
      c2Connections,
      dataExfiltration
    };
  }
  
  private analyzeFileSystemActivity(filesystem: any[]): {
    suspiciousModifications: number;
    systemFileAccess: boolean;
  } {
    const systemPaths = [
      'C:\\Windows\\System32',
      'C:\\Windows\\SysWOW64',
      'C:\\Program Files',
      '/etc',
      '/usr/bin',
      '/usr/sbin',
      '/System/Library',
      '/Library'
    ];
    
    const suspiciousModifications = filesystem.filter(fs =>
      fs.action === 'modified' || fs.action === 'created'
    ).length;
    
    const systemFileAccess = filesystem.some(fs =>
      systemPaths.some(path => fs.path && fs.path.includes(path))
    );
    
    return {
      suspiciousModifications,
      systemFileAccess
    };
  }
  
  private analyzeProcessActivity(processes: any[]): {
    suspiciousProcesses: number;
    processInjection: boolean;
  } {
    const suspiciousNames = [
      'cmd.exe',
      'powershell.exe',
      'wscript.exe',
      'cscript.exe',
      'mshta.exe',
      'rundll32.exe',
      'regsvr32.exe'
    ];
    
    const suspiciousProcesses = processes.filter(proc =>
      suspiciousNames.some(name => 
        proc.name && proc.name.toLowerCase().includes(name.toLowerCase())
      )
    ).length;
    
    // Process injection indicators (processes with unexpected parent PIDs)
    const processInjection = processes.some(proc =>
      proc.parentPid && proc.parentPid !== 1 && proc.parentPid !== 0 // Not root/system process
    );
    
    return {
      suspiciousProcesses,
      processInjection
    };
  }
  
  private isKnownC2Domain(domain: string): boolean {
    if (!domain) return false;
    
    // Check against known C2 domain list
    // In production, this would query the IOC database from threat-intel service
    return this.knownC2Domains.has(domain.toLowerCase());
  }
  
  /**
   * Update known C2 domains (would be called from threat intel sync)
   */
  updateKnownC2Domains(domains: string[]): void {
    this.knownC2Domains = new Set(domains.map(d => d.toLowerCase()));
  }
}
