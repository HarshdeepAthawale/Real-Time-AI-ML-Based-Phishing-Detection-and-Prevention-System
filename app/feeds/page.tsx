'use client'

import { useState } from 'react'
import { Rss } from 'lucide-react'
import Header from '@/components/header'
import Navigation from '@/components/navigation'
import { FeedList } from '@/components/feeds/feed-list'
import { FeedForm } from '@/components/feeds/feed-form'
import { FeedStatus } from '@/components/feeds/feed-status'
import { Button } from '@/components/ui/button'
import { Plus } from 'lucide-react'
import { ThreatFeed } from '@/lib/api/feeds'

export default function FeedsPage() {
  const [showForm, setShowForm] = useState(false)
  const [editingFeed, setEditingFeed] = useState<ThreatFeed | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  const handleEdit = (feed: ThreatFeed) => {
    setEditingFeed(feed)
    setShowForm(true)
  }

  const handleFormSuccess = () => {
    setShowForm(false)
    setEditingFeed(null)
    setRefreshKey((k) => k + 1)
  }

  const handleCancel = () => {
    setShowForm(false)
    setEditingFeed(null)
  }

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Navigation activeTab="feeds" setActiveTab={() => {}} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Rss className="w-8 h-8 text-primary" />
                <div>
                  <h1 className="text-3xl font-bold text-foreground">Threat Feeds</h1>
                  <p className="text-muted-foreground mt-1">
                    Manage threat intelligence feed integrations
                  </p>
                </div>
              </div>
              {!showForm && (
                <Button onClick={() => setShowForm(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Feed
                </Button>
              )}
            </div>

            {/* Feed Status */}
            <FeedStatus />

            {/* Feed Form or List */}
            {showForm ? (
              <FeedForm
                feed={editingFeed}
                onSuccess={handleFormSuccess}
                onCancel={handleCancel}
              />
            ) : (
              <FeedList key={refreshKey} onEdit={handleEdit} onRefresh={() => setRefreshKey((k) => k + 1)} />
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
