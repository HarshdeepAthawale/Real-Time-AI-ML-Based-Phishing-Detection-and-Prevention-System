'use client'

import { useState, useEffect, Suspense } from 'react'
import { useSearchParams } from 'next/navigation'
import { Shield } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import Header from '@/components/header'
import Navigation from '@/components/navigation'
import { EmailDetector } from '@/components/detection/email-detector'
import { URLDetector } from '@/components/detection/url-detector'
import { TextDetector } from '@/components/detection/text-detector'
import { Spinner } from '@/components/ui/loading'

function DetectionContent() {
  const searchParams = useSearchParams()
  const [activeTab, setActiveTab] = useState('email')

  useEffect(() => {
    const tab = searchParams?.get('tab')
    if (tab && ['email', 'url', 'text'].includes(tab)) {
      setActiveTab(tab)
    }
  }, [searchParams])

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <Shield className="w-8 h-8 text-primary" />
          <div>
            <h1 className="text-3xl font-bold text-foreground">Threat Detection</h1>
            <p className="text-muted-foreground mt-1">
              Analyze emails, URLs, and text content for phishing attempts
            </p>
          </div>
        </div>
      </div>

      {/* Detection Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="email">Email</TabsTrigger>
          <TabsTrigger value="url">URL</TabsTrigger>
          <TabsTrigger value="text">Text</TabsTrigger>
        </TabsList>

        <TabsContent value="email" className="mt-6">
          <EmailDetector />
        </TabsContent>

        <TabsContent value="url" className="mt-6">
          <URLDetector />
        </TabsContent>

        <TabsContent value="text" className="mt-6">
          <TextDetector />
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default function DetectionPage() {
  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Navigation activeTab="detection" setActiveTab={() => {}} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          <Suspense fallback={
            <div className="flex items-center justify-center h-full">
              <Spinner />
            </div>
          }>
            <DetectionContent />
          </Suspense>
        </div>
      </main>
    </div>
  )
}
