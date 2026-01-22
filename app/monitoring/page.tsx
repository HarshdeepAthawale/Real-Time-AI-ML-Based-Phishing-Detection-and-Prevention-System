'use client'

import Header from '@/components/header'
import Navigation from '@/components/navigation'
import RealtimeMonitor from '@/components/realtime-monitor'

export default function MonitoringPage() {
  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Navigation />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          <RealtimeMonitor />
        </div>
      </main>
    </div>
  )
}
