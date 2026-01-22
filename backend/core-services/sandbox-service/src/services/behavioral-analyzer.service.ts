import { logger } from '../utils/logger';

export interface BehavioralIndicators {
  overall_risk: number; // 0-100
  categories: {
    network: {
      score: number;
      indicators: string[];
      connections_count: number;
      suspicious_ips: string[];
      suspicious_domains: string[];
    };
    process: {
      score: number;
      indicators: string[];
      malicious_processes: string[];
      injection_detected: boolean;
    };
    filesystem: {
      score: number;
      indicators: string[];
      modifications: number;
      suspicious_locations: string[];
    };
    registry: {
      score: number;
      indicators: string[];
      modifications: number;
      persistence_mechanisms: string[];
    };
    evasion: {
      score: number;
      indicators: string[];
      techniques: string[];
    };
  };
  mitre_techniques: string[];
  threat_classification: 'malware' | 'phishing' | 'ransomware' | 'trojan' | 'adware' | 'unknown';
}

export class BehavioralAnalyzerService {
  /**
   * Analyze sandbox results for behavioral indicators
   */
  analyze(sandboxResult: any): BehavioralIndicators {
    logger.info('Analyzing behavioral indicators');

    const indicators: BehavioralIndicators = {
      overall_risk: 0,
      categories: {
        network: this.analyzeNetwork(sandboxResult.network),
        process: this.analyzeProcesses(sandboxResult.processes),
        filesystem: this.analyzeFilesystem(sandboxResult.files, sandboxResult.dropped),
        registry: this.analyzeRegistry(sandboxResult.registry),
        evasion: this.analyzeEvasion(sandboxResult)
      },
      mitre_techniques: this.extractMitreTechniques(sandboxResult.mitre || []),
      threat_classification: this.classifyThreat(sandboxResult)
    };

    // Calculate overall risk score
    indicators.overall_risk = this.calculateOverallRisk(indicators);

    return indicators;
  }

  /**
   * Analyze network behavior
   */
  private analyzeNetwork(network: any): BehavioralIndicators['categories']['network'] {
    const indicators: string[] = [];
    const suspiciousIPs: string[] = [];
    const suspiciousDomains: string[] = [];
    let score = 0;

    if (!network) {
      return {
        score: 0,
        indicators,
        connections_count: 0,
        suspicious_ips: [],
        suspicious_domains: []
      };
    }

    // Check connections
    const connections = network.connections || [];
    const hosts = network.hosts || [];
    const domains = network.domains || [];

    // High connection count is suspicious
    if (connections.length > 50) {
      indicators.push('Excessive network connections');
      score += 10;
    }

    // Check for connections to suspicious ports
    const suspiciousPorts = [4444, 5555, 6666, 8080, 8888, 9999, 31337];
    const suspiciousPortConnections = connections.filter((conn: any) =>
      suspiciousPorts.includes(conn.port || conn.dport)
    );
    if (suspiciousPortConnections.length > 0) {
      indicators.push(`Connections to suspicious ports: ${suspiciousPortConnections.map((c: any) => c.port || c.dport).join(', ')}`);
      score += 15;
    }

    // Check for C2 communication patterns
    if (this.detectC2Pattern(connections)) {
      indicators.push('Potential C2 communication pattern detected');
      score += 30;
    }

    // Check for TOR/VPN/Proxy usage
    const torExitNodes = this.checkTorExitNodes(hosts);
    if (torExitNodes.length > 0) {
      indicators.push('TOR exit node communication detected');
      suspiciousIPs.push(...torExitNodes);
      score += 20;
    }

    // Check for suspicious domains
    const suspiciousDomainPatterns = [
      /\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/, // IP-based domains
      /\.tk$|\.ml$|\.ga$|\.cf$|\.gq$/, // Free TLDs
      /^[a-z0-9]{10,}\./, // Long random subdomain
      /bit\.ly|tinyurl|shorturl/ // URL shorteners
    ];

    domains.forEach((domain: string) => {
      if (suspiciousDomainPatterns.some(pattern => pattern.test(domain))) {
        suspiciousDomains.push(domain);
        score += 5;
      }
    });

    if (suspiciousDomains.length > 0) {
      indicators.push(`Suspicious domains: ${suspiciousDomains.slice(0, 5).join(', ')}`);
    }

    // Check DNS queries
    if (network.dns && network.dns.length > 100) {
      indicators.push('Excessive DNS queries');
      score += 10;
    }

    // Check for data exfiltration
    const httpRequests = network.http || [];
    const largeUploads = httpRequests.filter((req: any) => 
      req.method === 'POST' && (req.size || 0) > 1000000
    );
    if (largeUploads.length > 0) {
      indicators.push('Potential data exfiltration via HTTP POST');
      score += 25;
    }

    return {
      score: Math.min(score, 100),
      indicators,
      connections_count: connections.length,
      suspicious_ips: suspiciousIPs,
      suspicious_domains: suspiciousDomains
    };
  }

  /**
   * Analyze process behavior
   */
  private analyzeProcesses(processes: any[]): BehavioralIndicators['categories']['process'] {
    const indicators: string[] = [];
    const maliciousProcesses: string[] = [];
    let score = 0;
    let injectionDetected = false;

    if (!processes || processes.length === 0) {
      return { score: 0, indicators, malicious_processes: [], injection_detected: false };
    }

    // Check for suspicious process names
    const suspiciousProcessNames = [
      'powershell.exe', 'cmd.exe', 'wscript.exe', 'cscript.exe',
      'regsvr32.exe', 'rundll32.exe', 'mshta.exe', 'certutil.exe'
    ];

    processes.forEach(proc => {
      const procName = (proc.process_name || proc.name || '').toLowerCase();
      const cmdLine = proc.command_line || proc.cmd || '';

      // Suspicious process names
      if (suspiciousProcessNames.includes(procName)) {
        indicators.push(`Suspicious process: ${procName}`);
        maliciousProcesses.push(procName);
        score += 10;
      }

      // Command line obfuscation
      if (this.detectObfuscation(cmdLine)) {
        indicators.push(`Obfuscated command line: ${procName}`);
        score += 15;
      }

      // Process injection
      if (proc.injected || this.detectInjection(proc)) {
        indicators.push(`Process injection detected: ${procName}`);
        injectionDetected = true;
        score += 30;
      }

      // Privilege escalation
      if (proc.elevated || cmdLine.includes('runas') || cmdLine.includes('UAC')) {
        indicators.push(`Privilege escalation attempt: ${procName}`);
        score += 20;
      }
    });

    // Check for process hollowing
    const hollowingDetected = processes.some(proc => 
      proc.hollowing || (proc.parent_id === 0 && proc.process_id !== 0)
    );
    if (hollowingDetected) {
      indicators.push('Process hollowing detected');
      score += 35;
    }

    return {
      score: Math.min(score, 100),
      indicators,
      malicious_processes: maliciousProcesses,
      injection_detected: injectionDetected
    };
  }

  /**
   * Analyze filesystem behavior
   */
  private analyzeFilesystem(files: any[], dropped: any[]): BehavioralIndicators['categories']['filesystem'] {
    const indicators: string[] = [];
    const suspiciousLocations: string[] = [];
    let score = 0;

    const allFiles = [...(files || []), ...(dropped || [])];

    if (allFiles.length === 0) {
      return { score: 0, indicators, modifications: 0, suspicious_locations: [] };
    }

    // Suspicious file locations
    const suspiciousPaths = [
      /\\AppData\\Roaming\\/i,
      /\\Temp\\/i,
      /\\ProgramData\\/i,
      /\\Users\\Public\\/i,
      /\\Windows\\System32\\/i,
      /\\Startup\\/i
    ];

    allFiles.forEach(file => {
      const path = file.path || '';

      suspiciousPaths.forEach(pattern => {
        if (pattern.test(path)) {
          suspiciousLocations.push(path);
          score += 5;
        }
      });

      // Executable in suspicious location
      if (path.match(/\.exe$|\.dll$|\.scr$/i) && suspiciousPaths.some(p => p.test(path))) {
        indicators.push(`Executable dropped in suspicious location: ${path}`);
        score += 15;
      }

      // Hidden or system files
      if (file.hidden || file.system) {
        indicators.push(`Hidden/system file created: ${file.name}`);
        score += 10;
      }
    });

    // Check for ransomware indicators
    const ransomwarePatterns = [
      /README|DECRYPT|RECOVERY|RESTORE/i,
      /\.(encrypted|locked|crypto|crypt)$/i
    ];

    const ransomwareFiles = allFiles.filter(file => 
      ransomwarePatterns.some(pattern => pattern.test(file.name || ''))
    );

    if (ransomwareFiles.length > 0) {
      indicators.push('Potential ransomware indicators detected');
      score += 40;
    }

    // Excessive file modifications
    if (allFiles.length > 50) {
      indicators.push(`Excessive file modifications: ${allFiles.length} files`);
      score += 15;
    }

    return {
      score: Math.min(score, 100),
      indicators,
      modifications: allFiles.length,
      suspicious_locations: [...new Set(suspiciousLocations)]
    };
  }

  /**
   * Analyze registry behavior
   */
  private analyzeRegistry(registry: any[]): BehavioralIndicators['categories']['registry'] {
    const indicators: string[] = [];
    const persistenceMechanisms: string[] = [];
    let score = 0;

    if (!registry || registry.length === 0) {
      return { score: 0, indicators, modifications: 0, persistence_mechanisms: [] };
    }

    // Persistence registry keys
    const persistenceKeys = [
      /\\Run$/i,
      /\\RunOnce$/i,
      /\\RunServices$/i,
      /\\Winlogon/i,
      /\\Services\\/i,
      /\\Shell\\\\Open\\\\Command/i
    ];

    registry.forEach(entry => {
      const key = entry.key || '';

      persistenceKeys.forEach(pattern => {
        if (pattern.test(key)) {
          persistenceMechanisms.push(key);
          indicators.push(`Persistence mechanism: ${key}`);
          score += 25;
        }
      });

      // Security settings modification
      if (key.match(/\\Policies\\|\\WindowsFirewall\\|\\WindowsDefender\\/i)) {
        indicators.push(`Security settings modified: ${key}`);
        score += 20;
      }

      // Browser modification
      if (key.match(/\\Software\\\\(Microsoft\\\\)?Internet Explorer|\\Chrome\\|\\Firefox\\/i)) {
        indicators.push('Browser settings modified');
        score += 10;
      }
    });

    // Excessive registry modifications
    if (registry.length > 50) {
      indicators.push(`Excessive registry modifications: ${registry.length}`);
      score += 15;
    }

    return {
      score: Math.min(score, 100),
      indicators,
      modifications: registry.length,
      persistence_mechanisms: [...new Set(persistenceMechanisms)]
    };
  }

  /**
   * Analyze evasion techniques
   */
  private analyzeEvasion(sandboxResult: any): BehavioralIndicators['categories']['evasion'] {
    const indicators: string[] = [];
    const techniques: string[] = [];
    let score = 0;

    // Check for anti-VM/sandbox
    if (sandboxResult.signatures) {
      const evasionSignatures = sandboxResult.signatures.filter((sig: any) =>
        sig.name.match(/anti.*vm|anti.*sandbox|detect.*virtual|evasion/i)
      );

      evasionSignatures.forEach((sig: any) => {
        techniques.push(sig.name);
        indicators.push(`Evasion technique: ${sig.name}`);
        score += 20;
      });
    }

    // Check for delays/sleep
    if (sandboxResult.behavioral?.delays || sandboxResult.behavioral?.sleep) {
      indicators.push('Execution delays detected (anti-analysis)');
      techniques.push('Sleep/Delay');
      score += 15;
    }

    // Check for environment checks
    if (sandboxResult.behavioral?.environment_checks) {
      indicators.push('Environment checks detected');
      techniques.push('Environment Detection');
      score += 15;
    }

    return {
      score: Math.min(score, 100),
      indicators,
      techniques: [...new Set(techniques)]
    };
  }

  /**
   * Extract MITRE ATT&CK techniques
   */
  private extractMitreTechniques(mitre: any[]): string[] {
    return mitre.map(m => `${m.id || m.technique}: ${m.tactic}`);
  }

  /**
   * Classify threat type
   */
  private classifyThreat(sandboxResult: any): BehavioralIndicators['threat_classification'] {
    const signatures = sandboxResult.signatures || [];
    const tags = sandboxResult.tags || [];

    // Check signatures and tags
    const classifiers = {
      ransomware: /ransomware|crypto|encrypt|decrypt/i,
      trojan: /trojan|backdoor|rat/i,
      phishing: /phish|credentials|keylog/i,
      adware: /adware|pup|unwanted/i,
      malware: /malware|malicious|virus|worm/i
    };

    const allText = [
      ...signatures.map((s: any) => s.name),
      ...tags,
      sandboxResult.malwareFamily || ''
    ].join(' ');

    for (const [type, pattern] of Object.entries(classifiers)) {
      if (pattern.test(allText)) {
        return type as any;
      }
    }

    return 'unknown';
  }

  /**
   * Calculate overall risk score
   */
  private calculateOverallRisk(indicators: BehavioralIndicators): number {
    const weights = {
      network: 0.25,
      process: 0.30,
      filesystem: 0.20,
      registry: 0.15,
      evasion: 0.10
    };

    let totalScore = 0;
    totalScore += indicators.categories.network.score * weights.network;
    totalScore += indicators.categories.process.score * weights.process;
    totalScore += indicators.categories.filesystem.score * weights.filesystem;
    totalScore += indicators.categories.registry.score * weights.registry;
    totalScore += indicators.categories.evasion.score * weights.evasion;

    return Math.round(totalScore);
  }

  /**
   * Helper: Detect C2 communication patterns
   */
  private detectC2Pattern(connections: any[]): boolean {
    if (!connections || connections.length === 0) return false;

    // Check for beaconing (regular intervals)
    const timestamps = connections.map(c => c.timestamp || 0).filter(t => t > 0).sort();
    if (timestamps.length < 3) return false;

    const intervals = [];
    for (let i = 1; i < timestamps.length; i++) {
      intervals.push(timestamps[i] - timestamps[i - 1]);
    }

    // Check if intervals are suspiciously regular
    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((sum, interval) => sum + Math.pow(interval - avgInterval, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);

    return stdDev < avgInterval * 0.1; // Low variance indicates beaconing
  }

  /**
   * Helper: Check for TOR exit nodes
   */
  private checkTorExitNodes(hosts: string[]): string[] {
    // Simple check for common TOR ranges (simplified)
    const torRanges = ['185.', '176.', '144.'];
    return hosts.filter(host => torRanges.some(range => host.startsWith(range)));
  }

  /**
   * Helper: Detect command line obfuscation
   */
  private detectObfuscation(cmdLine: string): boolean {
    const obfuscationPatterns = [
      /\^/g, // Caret obfuscation
      /-[eE][nN][cC]/i, // Base64 encoding
      /`/g, // Backtick obfuscation
      /\+\s*\+/g, // String concatenation
      /\[char\]/i, // PowerShell char casting
      /frombase64/i
    ];

    return obfuscationPatterns.some(pattern => pattern.test(cmdLine));
  }

  /**
   * Helper: Detect process injection
   */
  private detectInjection(proc: any): boolean {
    const injectionIndicators = [
      proc.remote_thread,
      proc.process_hollowing,
      proc.apc_injection,
      proc.dll_injection
    ];

    return injectionIndicators.some(indicator => indicator === true);
  }
}
