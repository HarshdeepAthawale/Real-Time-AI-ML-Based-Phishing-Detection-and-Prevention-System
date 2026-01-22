'use client'

import { useState, useEffect } from 'react'
import { Database, Globe, LinkIcon, AlertTriangle } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { getMaliciousDomains, getThreatPatterns, getIOCs, getThreatIntelligenceSummary } from '@/lib/api/threat-intel'
import { MaliciousDomain, ThreatPattern, IOC, ThreatIntelligenceSummary } from '@/lib/types/api'
import { TableSkeleton, Spinner } from '@/components/ui/loading'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ErrorBoundary } from '@/components/error-boundary'

export default function ThreatIntelligence() {
  const [maliciousDomains, setMaliciousDomains] = useState<MaliciousDomain[]>([])
  const [threatPatterns, setThreatPatterns] = useState<ThreatPattern[]>([])
  const [ioCs, setIOCs] = useState<IOC[]>([])
  const [summary, setSummary] = useState<ThreatIntelligenceSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        const [domains, patterns, iocs, summaryData] = await Promise.all([
          getMaliciousDomains(50, 0),
          getThreatPatterns(20),
          getIOCs(50, 0),
          getThreatIntelligenceSummary(),
        ])
        setMaliciousDomains(domains)
        setThreatPatterns(patterns)
        setIOCs(iocs)
        setSummary(summaryData)
      } catch (err: any) {
        setError(err.message || 'Failed to load threat intelligence data')
        console.error('Error fetching threat intelligence:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    // Refresh every 5 minutes
    const interval = setInterval(fetchData, 300000)
    return () => clearInterval(interval)
  }, [])

  return (
    <ErrorBoundary>
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
            {loading ? (
              <TableSkeleton rows={4} />
            ) : maliciousDomains.length === 0 ? (
              <p className="text-sm text-muted-foreground">No malicious domains found</p>
            ) : (
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
            )}
          </div>
        </TabsContent>

        {/* Threat Patterns Tab */}
        <TabsContent value="patterns" className="space-y-4">
          <div className="bg-card rounded-lg border border-border p-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">Common Attack Patterns</h3>
            {loading ? (
              <TableSkeleton rows={4} />
            ) : threatPatterns.length === 0 ? (
              <p className="text-sm text-muted-foreground">No threat patterns found</p>
            ) : (
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
            )}
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
                {loading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <tr key={i} className="border-b border-border">
                      <td className="px-6 py-4"><div className="h-4 w-32 bg-muted animate-pulse rounded" /></td>
                      <td className="px-6 py-4"><div className="h-6 w-20 bg-muted animate-pulse rounded" /></td>
                      <td className="px-6 py-4"><div className="h-4 w-16 bg-muted animate-pulse rounded" /></td>
                      <td className="px-6 py-4"><div className="h-4 w-12 bg-muted animate-pulse rounded" /></td>
                    </tr>
                  ))
                ) : ioCs.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-6 py-8 text-center text-sm text-muted-foreground">
                      No IOCs found
                    </td>
                  </tr>
                ) : (
                  ioCs.map((ioc) => (
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
                  ))
                )}
              </tbody>
            </table>
          </div>
        </TabsContent>
      </Tabs>

      {error && (
        <Alert>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Intelligence Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Known Threats</h4>
          {loading ? (
            <Spinner className="my-4" />
          ) : (
            <>
              <p className="text-3xl font-bold text-foreground">
                {summary ? (summary.knownThreats / 1000).toFixed(1) + 'K' : '0'}
              </p>
              <p className="text-xs text-muted-foreground mt-2">Active monitoring</p>
            </>
          )}
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Feed Integrations</h4>
          {loading ? (
            <Spinner className="my-4" />
          ) : (
            <>
              <p className="text-3xl font-bold text-foreground">{summary?.feedIntegrations || 0}</p>
              <p className="text-xs text-muted-foreground mt-2">Real-time updates</p>
            </>
          )}
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Zero-Day Detection</h4>
          {loading ? (
            <Spinner className="my-4" />
          ) : (
            <>
              <p className="text-3xl font-bold text-foreground">{summary?.zeroDayDetection || 0}%</p>
              <p className="text-xs text-muted-foreground mt-2">Success rate</p>
            </>
          )}
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Last Updated</h4>
          {loading ? (
            <Spinner className="my-4" />
          ) : (
            <>
              <p className="text-sm font-semibold text-foreground">
                {summary?.lastUpdated ? new Date(summary.lastUpdated).toLocaleTimeString() : 'Never'}
              </p>
              <p className="text-xs text-muted-foreground mt-2">Feeds synchronized</p>
            </>
          )}
        </div>
      </div>
    </div>
    </ErrorBoundary>
  )
}
