'use client'

import ThreatDashboard from '@/components/threat-dashboard'
import ThreatIntelligence from '@/components/threat-intelligence'
import RealtimeMonitor from '@/components/realtime-monitor'
import Header from '@/components/header'
import Navigation from '@/components/navigation'

export default function Page() {
  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Navigation />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          <ThreatDashboard />
        </div>
      </main>
    </div>
  )
}
