'use client'

import ThreatDashboard from '@/components/threat-dashboard'
import Header from '@/components/header'
import Navigation from '@/components/navigation'
import { ApiKeyBanner } from '@/components/api-key-banner'

export default function Page() {
  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Navigation />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          <div className="px-6 pt-6">
            <ApiKeyBanner />
          </div>
          <ThreatDashboard />
        </div>
      </main>
    </div>
  )
}
