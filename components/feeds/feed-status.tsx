'use client'

import { useState, useEffect } from 'react'
import { RefreshCw, CheckCircle2, Clock, AlertCircle } from 'lucide-react'
import { syncAllFeeds, syncFeed, getSyncStatus } from '@/lib/api/feeds'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Spinner } from '@/components/ui/loading'
import { useToast } from '@/hooks/use-toast'

export function FeedStatus() {
  const [syncing, setSyncing] = useState(false)
  const [status, setStatus] = useState<any>(null)
  const { toast } = useToast()

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchStatus = async () => {
    try {
      const data = await getSyncStatus()
      setStatus(data)
    } catch (error) {
      console.error('Error fetching sync status:', error)
    }
  }

  const handleSyncAll = async () => {
    try {
      setSyncing(true)
      await syncAllFeeds()
      toast({
        title: 'Sync Started',
        description: 'All feeds are being synchronized.',
      })
      setTimeout(fetchStatus, 2000)
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to sync feeds',
        variant: 'destructive',
      })
    } finally {
      setSyncing(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Sync Status</CardTitle>
            <CardDescription>Threat feed synchronization status</CardDescription>
          </div>
          <Button
            onClick={handleSyncAll}
            disabled={syncing}
            size="sm"
            variant="outline"
          >
            {syncing ? (
              <>
                <Spinner className="mr-2 h-4 w-4" />
                Syncing...
              </>
            ) : (
              <>
                <RefreshCw className="mr-2 h-4 w-4" />
                Sync All
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {status ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Last Sync</span>
              <span className="text-sm font-medium">
                {status.lastSyncAt
                  ? new Date(status.lastSyncAt).toLocaleString()
                  : 'Never'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Active Feeds</span>
              <Badge>{status.activeFeeds || 0}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Total Feeds</span>
              <Badge variant="outline">{status.totalFeeds || 0}</Badge>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center py-4">
            <Spinner />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
