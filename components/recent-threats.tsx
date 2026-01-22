'use client'

import { useState, useEffect } from 'react'
import { AlertTriangle, CheckCircle2, Clock, Shield } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { getRecentThreats } from '@/lib/api/dashboard'
import { Threat } from '@/lib/types/api'
import { ThreatRowSkeleton } from '@/components/ui/loading'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ErrorBoundary } from '@/components/error-boundary'

export default function RecentThreats() {
  const [threats, setThreats] = useState<Threat[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchThreats = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await getRecentThreats(10, 0)
      setThreats(data)
    } catch (err: any) {
      setError(err.message || 'Failed to load threats')
      console.error('Error fetching threats:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchThreats()
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchThreats, 30000)
    return () => clearInterval(interval)
  }, [])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-destructive/20 text-destructive border-destructive/30'
      case 'high':
        return 'bg-accent/20 text-accent border-accent/30'
      case 'medium':
        return 'bg-chart-3/20 text-chart-3 border-chart-3/30'
      default:
        return 'bg-chart-2/20 text-chart-2 border-chart-2/30'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'blocked':
        return <Shield className="w-4 h-4 text-chart-1" />
      case 'monitored':
        return <Clock className="w-4 h-4 text-chart-3" />
      case 'resolved':
        return <CheckCircle2 className="w-4 h-4 text-muted-foreground" />
      default:
        return null
    }
  }

  return (
    <ErrorBoundary>
      <div className="bg-card rounded-lg border border-border p-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">Recent Threats</h3>
      {error && (
        <Alert className="mb-4">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Type
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Target
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Severity
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Status
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Time
              </th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <ThreatRowSkeleton key={i} />
              ))
            ) : threats.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-sm text-muted-foreground">
                  No threats detected yet
                </td>
              </tr>
            ) : (
              threats.map((threat) => (
              <tr
                key={threat.id}
                className="border-b border-border hover:bg-muted/50 transition-colors"
              >
                <td className="px-4 py-4 text-sm text-foreground font-medium">{threat.type}</td>
                <td className="px-4 py-4 text-sm text-muted-foreground font-mono">{threat.target}</td>
                <td className="px-4 py-4 text-sm">
                  <Badge
                    variant="outline"
                    className={`capitalize ${getSeverityColor(threat.severity)}`}
                  >
                    {threat.severity}
                  </Badge>
                </td>
                <td className="px-4 py-4 text-sm">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(threat.status)}
                    <span className="capitalize text-muted-foreground">{threat.status}</span>
                  </div>
                </td>
                <td className="px-4 py-4 text-sm text-muted-foreground">{threat.timestamp}</td>
              </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
    </ErrorBoundary>
  )
}
