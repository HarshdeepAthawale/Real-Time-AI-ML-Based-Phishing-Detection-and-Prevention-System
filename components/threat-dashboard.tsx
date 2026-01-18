'use client'

import { AlertTriangle, CheckCircle2, Clock, TrendingUp } from 'lucide-react'
import StatCard from '@/components/stat-card'
import ThreatChart from '@/components/threat-chart'
import RecentThreats from '@/components/recent-threats'

export default function ThreatDashboard() {
  const stats = [
    {
      title: 'Critical Threats',
      value: '12',
      change: '+3 this hour',
      icon: AlertTriangle,
      color: 'text-destructive',
      bg: 'bg-destructive/10',
    },
    {
      title: 'Detection Rate',
      value: '98.7%',
      change: '+0.2% today',
      icon: CheckCircle2,
      color: 'text-chart-1',
      bg: 'bg-chart-1/10',
    },
    {
      title: 'Avg Response Time',
      value: '45ms',
      change: 'Within SLA',
      icon: Clock,
      color: 'text-chart-2',
      bg: 'bg-chart-2/10',
    },
    {
      title: 'Phishing Attempts',
      value: '2,847',
      change: 'Last 24h',
      icon: TrendingUp,
      color: 'text-accent',
      bg: 'bg-accent/10',
    },
  ]

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground">Security Dashboard</h1>
        <p className="text-muted-foreground mt-1">Real-time threat detection and analysis</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <StatCard key={stat.title} {...stat} />
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ThreatChart />
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h3 className="text-lg font-semibold text-foreground mb-4">Threat Distribution</h3>
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Email Phishing</span>
                <span className="text-foreground font-semibold">45%</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div className="h-full w-[45%] bg-chart-1 rounded-full" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">URL Spoofing</span>
                <span className="text-foreground font-semibold">30%</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div className="h-full w-[30%] bg-chart-2 rounded-full" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Domain Hijacking</span>
                <span className="text-foreground font-semibold">15%</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div className="h-full w-[15%] bg-chart-3 rounded-full" />
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">AI-Generated Content</span>
                <span className="text-foreground font-semibold">10%</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div className="h-full w-[10%] bg-accent rounded-full" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Threats */}
      <RecentThreats />
    </div>
  )
}
