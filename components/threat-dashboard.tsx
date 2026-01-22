'use client'

import { useState, useEffect } from 'react'
import { AlertTriangle, CheckCircle2, Clock, TrendingUp } from 'lucide-react'
import StatCard from '@/components/stat-card'
import ThreatChart from '@/components/threat-chart'
import RecentThreats from '@/components/recent-threats'
import { getDashboardStats, getThreatDistribution } from '@/lib/api/dashboard'
import { DashboardStats, ThreatDistribution } from '@/lib/types/api'
import { StatCardSkeleton } from '@/components/ui/loading'
import { ErrorBoundary } from '@/components/error-boundary'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { QuickActions } from '@/components/dashboard/quick-actions'

export default function ThreatDashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [distribution, setDistribution] = useState<ThreatDistribution | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        const [statsData, distributionData] = await Promise.all([
          getDashboardStats(),
          getThreatDistribution(),
        ])
        setStats(statsData)
        setDistribution(distributionData)
      } catch (err: any) {
        setError(err.message || 'Failed to load dashboard data')
        console.error('Error fetching dashboard data:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    // Refresh every 60 seconds
    const interval = setInterval(fetchData, 60000)
    return () => clearInterval(interval)
  }, [])

  const statCards = stats ? [
    {
      title: 'Critical Threats',
      value: stats.criticalThreats.toString(),
      change: stats.criticalThreatsChange || '',
      icon: AlertTriangle,
      color: 'text-destructive',
      bg: 'bg-destructive/10',
    },
    {
      title: 'Detection Rate',
      value: `${stats.detectionRate.toFixed(1)}%`,
      change: stats.detectionRateChange || '',
      icon: CheckCircle2,
      color: 'text-chart-1',
      bg: 'bg-chart-1/10',
    },
    {
      title: 'Avg Response Time',
      value: `${stats.avgResponseTime}ms`,
      change: stats.avgResponseTimeChange || '',
      icon: Clock,
      color: 'text-chart-2',
      bg: 'bg-chart-2/10',
    },
    {
      title: 'Phishing Attempts',
      value: stats.phishingAttempts.toLocaleString(),
      change: stats.phishingAttemptsPeriod || '',
      icon: TrendingUp,
      color: 'text-accent',
      bg: 'bg-accent/10',
    },
  ] : []

  return (
    <ErrorBoundary>
      <div className="p-6 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-foreground">Security Dashboard</h1>
          <p className="text-muted-foreground mt-1">Real-time threat detection and analysis</p>
        </div>

        {error && (
          <Alert>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Quick Actions */}
        <QuickActions />

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {loading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <StatCardSkeleton key={i} />
            ))
          ) : (
            statCards.map((stat) => (
              <StatCard key={stat.title} {...stat} />
            ))
          )}
        </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ThreatChart />
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4">Threat Distribution</h3>
          {loading ? (
            <div className="space-y-4">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="space-y-2">
                  <div className="h-2 bg-muted rounded-full animate-pulse" />
                </div>
              ))}
            </div>
          ) : distribution ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Email Phishing</span>
                  <span className="text-foreground font-semibold">{distribution.emailPhishing}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-chart-1 rounded-full" style={{ width: `${distribution.emailPhishing}%` }} />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">URL Spoofing</span>
                  <span className="text-foreground font-semibold">{distribution.urlSpoofing}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-chart-2 rounded-full" style={{ width: `${distribution.urlSpoofing}%` }} />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Domain Hijacking</span>
                  <span className="text-foreground font-semibold">{distribution.domainHijacking}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-chart-3 rounded-full" style={{ width: `${distribution.domainHijacking}%` }} />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">AI-Generated Content</span>
                  <span className="text-foreground font-semibold">{distribution.aiGeneratedContent}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-accent rounded-full" style={{ width: `${distribution.aiGeneratedContent}%` }} />
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No distribution data available</p>
          )}
        </div>
      </div>

      {/* Recent Threats */}
      <RecentThreats />
    </div>
    </ErrorBoundary>
  )
}
