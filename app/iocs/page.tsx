'use client'

import { useState, useEffect, Suspense } from 'react'
import { useSearchParams } from 'next/navigation'
import { Database } from 'lucide-react'
import Header from '@/components/header'
import Navigation from '@/components/navigation'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { IOCChecker } from '@/components/ioc/ioc-checker'
import { IOCBulkChecker } from '@/components/ioc/ioc-bulk-checker'
import { IOCSearch } from '@/components/ioc/ioc-search'
import { IOCReporter } from '@/components/ioc/ioc-reporter'
import { IOCStats } from '@/components/ioc/ioc-stats'
import { ApiKeyBanner } from '@/components/api-key-banner'
import { Spinner } from '@/components/ui/loading'

function IOCSContent() {
  const searchParams = useSearchParams()
  const [activeTab, setActiveTab] = useState('check')

  useEffect(() => {
    const tab = searchParams?.get('tab')
    if (tab && ['check', 'bulk', 'search', 'report'].includes(tab)) {
      setActiveTab(tab)
    }
  }, [searchParams])

  return (
    <div className="p-6 space-y-6">
      <ApiKeyBanner />
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <Database className="w-8 h-8 text-primary" />
          <div>
            <h1 className="text-3xl font-bold text-foreground">IOC Management</h1>
            <p className="text-muted-foreground mt-1">
              Check, search, and report Indicators of Compromise
            </p>
          </div>
        </div>
      </div>

      {/* Stats */}
      <IOCStats />

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="check">Check</TabsTrigger>
          <TabsTrigger value="bulk">Bulk Check</TabsTrigger>
          <TabsTrigger value="search">Search</TabsTrigger>
          <TabsTrigger value="report">Report</TabsTrigger>
        </TabsList>

        <TabsContent value="check" className="mt-6">
          <IOCChecker />
        </TabsContent>

        <TabsContent value="bulk" className="mt-6">
          <IOCBulkChecker />
        </TabsContent>

        <TabsContent value="search" className="mt-6">
          <IOCSearch />
        </TabsContent>

        <TabsContent value="report" className="mt-6">
          <IOCReporter />
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default function IOCSPage() {
  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Navigation activeTab="iocs" setActiveTab={() => {}} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          <Suspense fallback={
            <div className="flex items-center justify-center h-full">
              <Spinner />
            </div>
          }>
            <IOCSContent />
          </Suspense>
        </div>
      </main>
    </div>
  )
}
