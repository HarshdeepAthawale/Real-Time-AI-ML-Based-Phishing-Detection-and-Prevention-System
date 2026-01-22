'use client'

import { useState, useEffect } from 'react'
import { Database, AlertTriangle, TrendingUp } from 'lucide-react'
import { getIOCStats, IOCStats } from '@/lib/api/ioc'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Spinner } from '@/components/ui/loading'
import { Progress } from '@/components/ui/progress'

export function IOCStats() {
  const [stats, setStats] = useState<IOCStats | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true)
        const data = await getIOCStats()
        setStats(data)
      } catch (error) {
        console.error('Error fetching IOC stats:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
  }, [])

  if (loading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-center py-8">
            <Spinner />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!stats) {
    return null
  }

  const totalBySeverity = Object.values(stats.bySeverity || {}).reduce((a, b) => a + b, 0)
  const totalByType = Object.values(stats.byType || {}).reduce((a, b) => a + b, 0)

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-primary" />
            <CardTitle>Total IOCs</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-bold">{stats.total.toLocaleString()}</p>
          <p className="text-sm text-muted-foreground mt-2">In database</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-destructive" />
            <CardTitle>By Severity</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {Object.entries(stats.bySeverity || {}).map(([severity, count]) => (
            <div key={severity}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm capitalize">{severity}</span>
                <span className="text-sm font-medium">{count}</span>
              </div>
              <Progress
                value={totalBySeverity > 0 ? (count / totalBySeverity) * 100 : 0}
                className="h-2"
              />
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-chart-1" />
            <CardTitle>By Type</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {Object.entries(stats.byType || {})
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5)
            .map(([type, count]) => (
              <div key={type}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm capitalize">{type.replace('_', ' ')}</span>
                  <span className="text-sm font-medium">{count}</span>
                </div>
                <Progress
                  value={totalByType > 0 ? (count / totalByType) * 100 : 0}
                  className="h-2"
                />
              </div>
            ))}
        </CardContent>
      </Card>
    </div>
  )
}
