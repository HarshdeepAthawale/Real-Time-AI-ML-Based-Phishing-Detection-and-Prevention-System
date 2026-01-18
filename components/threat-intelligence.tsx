import { Database, Globe, LinkIcon, AlertTriangle } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'

export default function ThreatIntelligence() {
  const maliciousDomains = [
    { domain: 'attacker-bank-clone.io', reputation: 'Malicious', reports: 342 },
    { domain: 'verify-account-now.fake', reputation: 'Malicious', reports: 156 },
    { domain: 'secure-login-service.co', reputation: 'Suspicious', reports: 89 },
    { domain: 'update-credentials-here.net', reputation: 'Malicious', reports: 267 },
  ]

  const threatPatterns = [
    { pattern: 'WHOIS spoofing with lookalike registrars', incidents: 1240 },
    { pattern: 'Homoglyph domain attacks (0/O, 1/l)', incidents: 856 },
    { pattern: 'SSL certificate misuse (free providers)', incidents: 623 },
    { pattern: 'URL parameter cloaking with redirects', incidents: 445 },
  ]

  const ioCs = [
    { value: '192.168.1.100', type: 'IP Address', sources: 45 },
    { value: 'hash:a1b2c3d4e5f6', type: 'File Hash', sources: 12 },
    { value: 'C2-command.malware.net', type: 'Domain', sources: 67 },
    { value: 'payload_dropper_v3.exe', type: 'Filename', sources: 23 },
  ]

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Threat Intelligence</h1>
        <p className="text-muted-foreground mt-1">Integrated threat feeds and indicators of compromise</p>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="domains" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-muted">
          <TabsTrigger value="domains" className="gap-2">
            <Globe className="w-4 h-4" />
            Malicious Domains
          </TabsTrigger>
          <TabsTrigger value="patterns" className="gap-2">
            <AlertTriangle className="w-4 h-4" />
            Threat Patterns
          </TabsTrigger>
          <TabsTrigger value="iocs" className="gap-2">
            <Database className="w-4 h-4" />
            IOCs
          </TabsTrigger>
        </TabsList>

        {/* Malicious Domains Tab */}
        <TabsContent value="domains" className="space-y-4">
          <div className="bg-card rounded-lg border border-border p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">Recently Identified Domains</h3>
            <div className="space-y-3">
              {maliciousDomains.map((item) => (
                <div key={item.domain} className="p-4 bg-muted/50 rounded-lg border border-border">
                  <div className="flex items-center justify-between mb-2">
                    <code className="text-sm font-mono text-foreground">{item.domain}</code>
                    <Badge
                      variant="outline"
                      className={
                        item.reputation === 'Malicious'
                          ? 'bg-destructive/20 text-destructive border-destructive/30'
                          : 'bg-accent/20 text-accent border-accent/30'
                      }
                    >
                      {item.reputation}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">{item.reports} reports from threat feeds</p>
                </div>
              ))}
            </div>
          </div>
        </TabsContent>

        {/* Threat Patterns Tab */}
        <TabsContent value="patterns" className="space-y-4">
          <div className="bg-card rounded-lg border border-border p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">Common Attack Patterns</h3>
            <div className="space-y-3">
              {threatPatterns.map((pattern) => (
                <div
                  key={pattern.pattern}
                  className="p-4 bg-muted/50 rounded-lg border border-border hover:border-primary/50 transition-colors"
                >
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-medium text-foreground">{pattern.pattern}</p>
                    <span className="text-xs font-semibold text-accent bg-accent/20 px-2 py-1 rounded">
                      {pattern.incidents} incidents
                    </span>
                  </div>
                  <div className="w-full h-1 bg-border rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-primary to-accent"
                      style={{
                        width: `${Math.min(100, (pattern.incidents / 1240) * 100)}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </TabsContent>

        {/* IOCs Tab */}
        <TabsContent value="iocs" className="space-y-4">
          <div className="bg-card rounded-lg border border-border overflow-hidden">
            <table className="w-full">
              <thead className="bg-muted/50 border-b border-border">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground">
                    Indicator
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground">
                    Sources
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-muted-foreground">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody>
                {ioCs.map((ioc) => (
                  <tr key={ioc.value} className="border-b border-border hover:bg-muted/50 transition-colors">
                    <td className="px-6 py-4">
                      <code className="text-sm font-mono text-foreground break-all">{ioc.value}</code>
                    </td>
                    <td className="px-6 py-4">
                      <Badge variant="secondary">{ioc.type}</Badge>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-sm text-muted-foreground">{ioc.sources} sources</span>
                    </td>
                    <td className="px-6 py-4">
                      <button className="text-xs font-semibold text-primary hover:text-primary/80 transition-colors">
                        Block
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </TabsContent>
      </Tabs>

      {/* Intelligence Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Known Threats</h4>
          <p className="text-3xl font-bold text-foreground">8.4K</p>
          <p className="text-xs text-muted-foreground mt-2">Active monitoring</p>
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Feed Integrations</h4>
          <p className="text-3xl font-bold text-foreground">12</p>
          <p className="text-xs text-muted-foreground mt-2">Real-time updates</p>
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Zero-Day Detection</h4>
          <p className="text-3xl font-bold text-foreground">97%</p>
          <p className="text-xs text-muted-foreground mt-2">Success rate</p>
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Last Updated</h4>
          <p className="text-sm font-semibold text-foreground">2 minutes ago</p>
          <p className="text-xs text-muted-foreground mt-2">Feeds synchronized</p>
        </div>
      </div>
    </div>
  )
}
