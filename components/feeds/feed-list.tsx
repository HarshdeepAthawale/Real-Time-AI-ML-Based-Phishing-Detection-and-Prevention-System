'use client'

import { useState, useEffect } from 'react'
import { RefreshCw, Trash2, Edit, Power, PowerOff } from 'lucide-react'
import { listFeeds, deleteFeed, toggleFeed, ThreatFeed } from '@/lib/api/feeds'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { TableSkeleton } from '@/components/ui/loading'
import { formatDistanceToNow } from 'date-fns'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { useToast } from '@/hooks/use-toast'

interface FeedListProps {
  onEdit: (feed: ThreatFeed) => void
  onRefresh: () => void
}

export function FeedList({ onEdit, onRefresh }: FeedListProps) {
  const [feeds, setFeeds] = useState<ThreatFeed[]>([])
  const [loading, setLoading] = useState(true)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [feedToDelete, setFeedToDelete] = useState<ThreatFeed | null>(null)
  const { toast } = useToast()

  useEffect(() => {
    fetchFeeds()
  }, [])

  const fetchFeeds = async () => {
    try {
      setLoading(true)
      const response = await listFeeds()
      setFeeds(response.feeds)
    } catch (error) {
      console.error('Error fetching feeds:', error)
      toast({
        title: 'Error',
        description: 'Failed to load threat feeds',
        variant: 'destructive',
      })
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async () => {
    if (!feedToDelete) return

    try {
      await deleteFeed(feedToDelete.id)
      toast({
        title: 'Feed Deleted',
        description: `Feed "${feedToDelete.name}" has been deleted.`,
      })
      fetchFeeds()
      onRefresh()
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to delete feed',
        variant: 'destructive',
      })
    } finally {
      setDeleteDialogOpen(false)
      setFeedToDelete(null)
    }
  }

  const handleToggle = async (feed: ThreatFeed) => {
    try {
      await toggleFeed(feed.id)
      toast({
        title: 'Feed Updated',
        description: `Feed "${feed.name}" has been ${feed.isActive ? 'deactivated' : 'activated'}.`,
      })
      fetchFeeds()
      onRefresh()
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to toggle feed',
        variant: 'destructive',
      })
    }
  }

  if (loading) {
    return <TableSkeleton rows={5} />
  }

  if (feeds.length === 0) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-muted-foreground text-center py-8">
            No threat feeds configured. Create one to get started.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
      <div className="space-y-2">
        {feeds.map((feed) => (
          <Card key={feed.id}>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-semibold">{feed.name}</h3>
                    <Badge variant="outline" className="capitalize">
                      {feed.feedType}
                    </Badge>
                    <Badge className={feed.isActive ? 'bg-chart-1' : 'bg-muted'}>
                      {feed.isActive ? 'Active' : 'Inactive'}
                    </Badge>
                  </div>
                  <div className="text-sm text-muted-foreground space-y-1">
                    {feed.apiEndpoint && (
                      <p className="font-mono text-xs truncate">{feed.apiEndpoint}</p>
                    )}
                    {feed.lastSyncAt && (
                      <p>
                        Last synced:{' '}
                        {formatDistanceToNow(new Date(feed.lastSyncAt), { addSuffix: true })}
                      </p>
                    )}
                    {feed.syncIntervalMinutes && (
                      <p>Sync interval: {feed.syncIntervalMinutes} minutes</p>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => handleToggle(feed)}
                    title={feed.isActive ? 'Deactivate' : 'Activate'}
                  >
                    {feed.isActive ? (
                      <PowerOff className="w-4 h-4" />
                    ) : (
                      <Power className="w-4 h-4" />
                    )}
                  </Button>
                  <Button variant="ghost" size="icon" onClick={() => onEdit(feed)} title="Edit">
                    <Edit className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      setFeedToDelete(feed)
                      setDeleteDialogOpen(true)
                    }}
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4 text-destructive" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Feed</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{feedToDelete?.name}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
