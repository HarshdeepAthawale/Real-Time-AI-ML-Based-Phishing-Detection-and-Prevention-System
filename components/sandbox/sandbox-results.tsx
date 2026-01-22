'use client'

import { SandboxAnalysis } from '@/lib/api/sandbox'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AlertTriangle, Info } from 'lucide-react'

interface SandboxResultsProps {
  analysis: SandboxAnalysis
}

export function SandboxResults({ analysis }: SandboxResultsProps) {
  if (!analysis.results) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-muted-foreground">No results available yet.</p>
        </CardContent>
      </Card>
    )
  }

  const results = analysis.results

  return (
    <Card>
      <CardHeader>
        <CardTitle>Analysis Results</CardTitle>
        <CardDescription>
          Detailed behavioral analysis from {analysis.sandbox_provider || 'sandbox'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="summary" className="w-full">
          <TabsList>
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="network">Network</TabsTrigger>
            <TabsTrigger value="filesystem">File System</TabsTrigger>
            <TabsTrigger value="processes">Processes</TabsTrigger>
          </TabsList>

          <TabsContent value="summary" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-muted/50 rounded-lg">
                <p className="text-sm text-muted-foreground mb-1">Status</p>
                <Badge>{analysis.status}</Badge>
              </div>
              <div className="p-4 bg-muted/50 rounded-lg">
                <p className="text-sm text-muted-foreground mb-1">Duration</p>
                <p className="font-medium">
                  {analysis.completed_at && analysis.started_at
                    ? `${Math.round(
                        (new Date(analysis.completed_at).getTime() -
                          new Date(analysis.started_at).getTime()) /
                          1000
                      )}s`
                    : 'N/A'}
                </p>
              </div>
            </div>

            {results.summary && (
              <div className="space-y-2">
                <h4 className="font-semibold">Summary</h4>
                <p className="text-sm text-muted-foreground">{results.summary}</p>
              </div>
            )}

            {results.malicious && (
              <div className="flex items-start gap-2 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                <AlertTriangle className="w-5 h-5 text-destructive mt-0.5" />
                <div>
                  <p className="font-semibold text-destructive">Malicious Activity Detected</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    This analysis has identified potentially malicious behavior.
                  </p>
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="network" className="space-y-4">
            {results.network ? (
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Network Connections</h4>
                  <div className="space-y-2">
                    {Array.isArray(results.network.connections) ? (
                      results.network.connections.map((conn: any, idx: number) => (
                        <div key={idx} className="p-3 bg-muted/50 rounded-lg text-sm">
                          <p className="font-mono">{conn.destination || conn.ip}</p>
                          {conn.port && <p className="text-muted-foreground">Port: {conn.port}</p>}
                        </div>
                      ))
                    ) : (
                      <pre className="p-4 bg-muted/50 rounded-lg text-xs overflow-auto">
                        {JSON.stringify(results.network, null, 2)}
                      </pre>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Info className="w-4 h-4" />
                <p className="text-sm">No network activity recorded</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="filesystem" className="space-y-4">
            {results.filesystem ? (
              <div className="space-y-2">
                {Array.isArray(results.filesystem) ? (
                  results.filesystem.map((file: any, idx: number) => (
                    <div key={idx} className="p-3 bg-muted/50 rounded-lg text-sm">
                      <p className="font-mono">{file.path || file}</p>
                    </div>
                  ))
                ) : (
                  <pre className="p-4 bg-muted/50 rounded-lg text-xs overflow-auto">
                    {JSON.stringify(results.filesystem, null, 2)}
                  </pre>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Info className="w-4 h-4" />
                <p className="text-sm">No file system activity recorded</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="processes" className="space-y-4">
            {results.processes ? (
              <div className="space-y-2">
                {Array.isArray(results.processes) ? (
                  results.processes.map((proc: any, idx: number) => (
                    <div key={idx} className="p-3 bg-muted/50 rounded-lg text-sm">
                      <p className="font-medium">{proc.name || proc}</p>
                      {proc.pid && <p className="text-muted-foreground">PID: {proc.pid}</p>}
                    </div>
                  ))
                ) : (
                  <pre className="p-4 bg-muted/50 rounded-lg text-xs overflow-auto">
                    {JSON.stringify(results.processes, null, 2)}
                  </pre>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Info className="w-4 h-4" />
                <p className="text-sm">No process activity recorded</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
