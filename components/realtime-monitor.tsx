'use client'

import { useState, useEffect } from 'react'
import { Activity, Zap, AlertCircle, Eye } from 'lucide-react'

interface LiveEvent {
  id: string
  type: 'detection' | 'blocked' | 'alert'
  message: string
  timestamp: Date
}

export default function RealtimeMonitor() {
  const [events, setEvents] = useState<LiveEvent[]>([
    {
      id: '1',
      type: 'blocked',
      message: 'Phishing email blocked from external sender',
      timestamp: new Date(Date.now() - 5000),
    },
    {
      id: '2',
      type: 'detection',
      message: 'Suspicious URL pattern detected in message',
      timestamp: new Date(Date.now() - 15000),
    },
    {
      id: '3',
      type: 'alert',
      message: 'High-confidence AI-generated content identified',
      timestamp: new Date(Date.now() - 45000),
    },
  ])

  useEffect(() => {
    const interval = setInterval(() => {
      const types: ('detection' | 'blocked' | 'alert')[] = ['detection', 'blocked', 'alert']
      const messages = {
        detection: [
          'Suspicious URL pattern detected',
          'Homoglyph attack indicators found',
          'Domain reputation score lowered',
        ],
        blocked: [
          'Phishing email blocked',
          'Malicious link intercepted',
          'Credential harvesting attempt blocked',
        ],
        alert: [
          'AI-generated content identified',
          'New threat pattern detected',
          'Anomaly in email headers',
        ],
      }

      const randomType = types[Math.floor(Math.random() * types.length)]
      const randomMessage =
        messages[randomType][Math.floor(Math.random() * messages[randomType].length)]

      const newEvent: LiveEvent = {
        id: Date.now().toString(),
        type: randomType,
        message: randomMessage,
        timestamp: new Date(),
      }

      setEvents((prev) => [newEvent, ...prev.slice(0, 19)])
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const getEventColor = (type: string) => {
    switch (type) {
      case 'blocked':
        return 'border-chart-1 bg-chart-1/5'
      case 'detection':
        return 'border-chart-2 bg-chart-2/5'
      case 'alert':
        return 'border-accent bg-accent/5'
      default:
        return 'border-border bg-muted/5'
    }
  }

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'blocked':
        return <Zap className="w-4 h-4 text-chart-1" />
      case 'detection':
        return <Eye className="w-4 h-4 text-chart-2" />
      case 'alert':
        return <AlertCircle className="w-4 h-4 text-accent" />
      default:
        return <Activity className="w-4 h-4 text-primary" />
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-foreground flex items-center gap-2">
          <Activity className="w-8 h-8 text-primary animate-pulse" />
          Real-Time Monitoring
        </h1>
        <p className="text-muted-foreground mt-1">Live threat event stream</p>
      </div>

      {/* Status Indicator */}
      <div className="bg-card rounded-lg border border-border p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-chart-1 rounded-full animate-pulse" />
            <div>
              <h3 className="text-lg font-semibold text-foreground">System Status</h3>
              <p className="text-sm text-muted-foreground">All systems operational</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-foreground">15.2k</p>
            <p className="text-xs text-muted-foreground">events/hour</p>
          </div>
        </div>
      </div>

      {/* Events Log */}
      <div className="bg-card rounded-lg border border-border p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Event Stream</h3>
        <div className="space-y-3 max-h-[600px] overflow-y-auto">
          {events.map((event) => (
            <div
              key={event.id}
              className={`p-4 rounded-lg border transition-all ${getEventColor(event.type)}`}
            >
              <div className="flex items-start gap-3">
                <div className="mt-1">{getEventIcon(event.type)}</div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground capitalize">{event.type}</p>
                  <p className="text-sm text-muted-foreground mt-1">{event.message}</p>
                </div>
                <div className="text-xs text-muted-foreground whitespace-nowrap ml-4">
                  {Math.round((Date.now() - event.timestamp.getTime()) / 1000)}s ago
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Network Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Active Connections</h4>
          <p className="text-3xl font-bold text-foreground">1,247</p>
          <p className="text-xs text-muted-foreground mt-2">+89 since last hour</p>
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Traffic Scanned</h4>
          <p className="text-3xl font-bold text-foreground">2.3GB</p>
          <p className="text-xs text-muted-foreground mt-2">Real-time analysis</p>
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Model Accuracy</h4>
          <p className="text-3xl font-bold text-foreground">98.7%</p>
          <p className="text-xs text-muted-foreground mt-2">Last 24 hours</p>
        </div>
      </div>
    </div>
  )
}
