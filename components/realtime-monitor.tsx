'use client'

import { useState, useEffect } from 'react'
import { Activity, Zap, AlertCircle, Eye, Wifi, WifiOff } from 'lucide-react'
import { useWebSocket } from '@/hooks/use-websocket'
import { LiveEvent } from '@/lib/types/api'
import { Spinner } from '@/components/ui/loading'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ErrorBoundary } from '@/components/error-boundary'

export default function RealtimeMonitor() {
  const [apiKey, setApiKey] = useState<string | undefined>()
  const [organizationId, setOrganizationId] = useState<string | undefined>()

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setApiKey(localStorage.getItem('api_key') || undefined)
      setOrganizationId(localStorage.getItem('organization_id') || undefined)
    }
  }, [])

  const { isConnected, events, connect, disconnect } = useWebSocket({
    autoConnect: true,
    apiKey: apiKey || undefined,
    organizationId: organizationId || undefined,
    onConnect: () => {
      if (process.env.NODE_ENV === 'development') console.log('WebSocket connected')
    },
    onDisconnect: () => {
      if (process.env.NODE_ENV === 'development') console.log('WebSocket disconnected')
    },
    onError: (error) => {
      console.error('WebSocket error:', error)
    },
  })

  // Format events for display
  const displayEvents: LiveEvent[] = events.map((event) => ({
    id: event.id,
    type: event.type === 'threat_detected' ? 'alert' : 
          event.type === 'url_analyzed' || event.type === 'email_analyzed' ? 'detection' : 
          event.type as 'detection' | 'blocked' | 'alert',
    message: event.message,
    timestamp: event.timestamp,
  }));

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
    <ErrorBoundary>
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
            {isConnected ? (
              <Wifi className="w-5 h-5 text-chart-1" />
            ) : (
              <WifiOff className="w-5 h-5 text-destructive" />
            )}
            <div>
              <h3 className="text-lg font-semibold text-foreground">System Status</h3>
              <p className="text-sm text-muted-foreground">
                {isConnected ? 'All systems operational' : 'Connecting...'}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-foreground">{events.length}</p>
            <p className="text-xs text-muted-foreground">events received</p>
          </div>
        </div>
        {!isConnected && (
          <Alert className="mt-4" variant="destructive">
            <AlertDescription>
              WebSocket disconnected. Attempting to reconnect... Ensure API Base URL is http://localhost:3000 and a valid API key is saved in Settings (e.g. testkey_smoke_test_12345 for local dev).
            </AlertDescription>
          </Alert>
        )}
      </div>

      {/* Events Log */}
      <div className="bg-card rounded-lg border border-border p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Event Stream</h3>
        {displayEvents.length === 0 && !isConnected ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <Spinner className="mb-4" />
              <p className="text-sm text-muted-foreground">Connecting to event stream...</p>
            </div>
          </div>
        ) : displayEvents.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-sm text-muted-foreground">No events yet. Waiting for threats...</p>
          </div>
        ) : (
          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            {displayEvents.map((event) => (
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
                  {(() => {
                    const timestamp = event.timestamp instanceof Date ? event.timestamp : new Date(event.timestamp);
                    const seconds = Math.round((Date.now() - timestamp.getTime()) / 1000);
                    if (seconds < 60) return `${seconds}s ago`;
                    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
                    return `${Math.floor(seconds / 3600)}h ago`;
                  })()}
                </div>
              </div>
            </div>
            ))}
          </div>
        )}
      </div>

      {/* Network Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Events Received</h4>
          <p className="text-3xl font-bold text-foreground">{events.length}</p>
          <p className="text-xs text-muted-foreground mt-2">Real-time stream</p>
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Connection Status</h4>
          <p className={`text-3xl font-bold ${isConnected ? 'text-chart-1' : 'text-destructive'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </p>
          <p className="text-xs text-muted-foreground mt-2">
            {isConnected ? 'WebSocket active' : 'Reconnecting...'}
          </p>
        </div>
        <div className="bg-card rounded-lg border border-border p-6">
          <h4 className="text-sm font-semibold text-muted-foreground mb-2">Threats Detected</h4>
          <p className="text-3xl font-bold text-foreground">
            {displayEvents.filter(e => e.type === 'alert' || e.type === 'blocked').length}
          </p>
          <p className="text-xs text-muted-foreground mt-2">In current session</p>
        </div>
      </div>
    </div>
    </ErrorBoundary>
  )
}
