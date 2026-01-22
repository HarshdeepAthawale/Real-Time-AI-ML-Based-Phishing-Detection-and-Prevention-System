'use client'

import { useState, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { createFeed, updateFeed, CreateFeedRequest, ThreatFeed } from '@/lib/api/feeds'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Spinner } from '@/components/ui/loading'
import { useToast } from '@/hooks/use-toast'
import { Switch } from '@/components/ui/switch'

const feedSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  feedType: z.enum(['misp', 'otx', 'custom', 'user_submitted']),
  apiEndpoint: z.string().url().optional().or(z.literal('')),
  apiKeyEncrypted: z.string().optional(),
  syncIntervalMinutes: z.number().int().min(1).max(10080).optional(),
  isActive: z.boolean().optional(),
})

type FeedFormData = z.infer<typeof feedSchema>

interface FeedFormProps {
  feed?: ThreatFeed | null
  onSuccess: () => void
  onCancel: () => void
}

export function FeedForm({ feed, onSuccess, onCancel }: FeedFormProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { toast } = useToast()

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue,
    reset,
  } = useForm<FeedFormData>({
    resolver: zodResolver(feedSchema),
    defaultValues: feed
      ? {
          name: feed.name,
          feedType: feed.feedType,
          apiEndpoint: feed.apiEndpoint || '',
          syncIntervalMinutes: feed.syncIntervalMinutes,
          isActive: feed.isActive,
        }
      : {
          feedType: 'custom',
          isActive: true,
          syncIntervalMinutes: 60,
        },
  })

  useEffect(() => {
    if (feed) {
      reset({
        name: feed.name,
        feedType: feed.feedType,
        apiEndpoint: feed.apiEndpoint || '',
        syncIntervalMinutes: feed.syncIntervalMinutes,
        isActive: feed.isActive,
      })
    }
  }, [feed, reset])

  const onSubmit = async (data: FeedFormData) => {
    try {
      setLoading(true)
      setError(null)

      const request: CreateFeedRequest = {
        name: data.name,
        feedType: data.feedType,
        apiEndpoint: data.apiEndpoint || undefined,
        syncIntervalMinutes: data.syncIntervalMinutes,
        isActive: data.isActive,
      }

      if (feed) {
        await updateFeed(feed.id, request)
        toast({
          title: 'Feed Updated',
          description: `Feed "${data.name}" has been updated.`,
        })
      } else {
        await createFeed(request)
        toast({
          title: 'Feed Created',
          description: `Feed "${data.name}" has been created.`,
        })
      }

      onSuccess()
    } catch (err: any) {
      setError(err.message || 'Failed to save feed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{feed ? 'Edit Feed' : 'Create Feed'}</CardTitle>
        <CardDescription>
          {feed ? 'Update threat feed configuration' : 'Add a new threat intelligence feed'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Feed Name *</Label>
            <Input id="name" {...register('name')} placeholder="My Threat Feed" />
            {errors.name && (
              <p className="text-sm text-destructive">{errors.name.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="feedType">Feed Type *</Label>
            <Select
              value={watch('feedType')}
              onValueChange={(value) => setValue('feedType', value as any)}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="misp">MISP</SelectItem>
                <SelectItem value="otx">AlienVault OTX</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
                <SelectItem value="user_submitted">User Submitted</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="apiEndpoint">API Endpoint (Optional)</Label>
            <Input
              id="apiEndpoint"
              type="url"
              placeholder="https://api.example.com/feeds"
              {...register('apiEndpoint')}
            />
            {errors.apiEndpoint && (
              <p className="text-sm text-destructive">{errors.apiEndpoint.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="syncIntervalMinutes">Sync Interval (minutes)</Label>
            <Input
              id="syncIntervalMinutes"
              type="number"
              min="1"
              max="10080"
              placeholder="60"
              {...register('syncIntervalMinutes', { valueAsNumber: true })}
            />
            <p className="text-xs text-muted-foreground">
              How often to sync this feed (1-10080 minutes)
            </p>
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="isActive">Active</Label>
              <p className="text-xs text-muted-foreground">
                Enable or disable this feed
              </p>
            </div>
            <Switch
              id="isActive"
              checked={watch('isActive')}
              onCheckedChange={(checked) => setValue('isActive', checked)}
            />
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="flex gap-2">
            <Button type="submit" disabled={loading} className="flex-1">
              {loading ? (
                <>
                  <Spinner className="mr-2 h-4 w-4" />
                  Saving...
                </>
              ) : (
                feed ? 'Update Feed' : 'Create Feed'
              )}
            </Button>
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
