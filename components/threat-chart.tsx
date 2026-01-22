'use client'

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { getChartData } from '@/lib/api/dashboard'
import { ChartDataPoint } from '@/lib/types/api'
import { ChartSkeleton } from '@/components/ui/loading'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ErrorBoundary } from '@/components/error-boundary'

export default function ThreatChart() {
  const [data, setData] = useState<ChartDataPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchChartData = async () => {
      try {
        setLoading(true)
        setError(null)
        const chartData = await getChartData(24)
        setData(chartData)
      } catch (err: any) {
        setError(err.message || 'Failed to load chart data')
        console.error('Error fetching chart data:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchChartData()
    // Refresh every 5 minutes
    const interval = setInterval(fetchChartData, 300000)
    return () => clearInterval(interval)
  }, [])

  return (
    <ErrorBoundary>
      <div className="bg-card rounded-lg border border-border p-6">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-foreground">Threat Detection Timeline</h3>
        <p className="text-sm text-muted-foreground">Last 24 hours</p>
      </div>

      {error && (
        <Alert className="mb-4">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {loading ? (
        <ChartSkeleton />
      ) : data.length === 0 ? (
        <div className="h-64 flex items-center justify-center">
          <p className="text-sm text-muted-foreground">No chart data available</p>
        </div>
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="time" stroke="var(--color-muted-foreground)" />
            <YAxis stroke="var(--color-muted-foreground)" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-card)',
                border: `1px solid var(--color-border)`,
                borderRadius: '8px',
              }}
              labelStyle={{ color: 'var(--color-foreground)' }}
              formatter={(value) => [`${value} threats`, 'Detected']}
            />
            <Line
              type="monotone"
              dataKey="threats"
              stroke="var(--color-primary)"
              strokeWidth={2}
              dot={{ fill: 'var(--color-primary)', r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      )}
    </div>
    </ErrorBoundary>
  )
}
