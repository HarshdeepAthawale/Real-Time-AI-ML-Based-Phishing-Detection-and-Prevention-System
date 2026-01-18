'use client'

import { useState } from 'react'
import { Shield, AlertTriangle, CheckCircle2, Clock, TrendingUp, Eye } from 'lucide-react'
import ThreatDashboard from '@/components/threat-dashboard'
import ThreatIntelligence from '@/components/threat-intelligence'
import RealtimeMonitor from '@/components/realtime-monitor'
import Header from '@/components/header'
import Navigation from '@/components/navigation'

export default function Page() {
  const [activeTab, setActiveTab] = useState('dashboard')

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          {activeTab === 'dashboard' && <ThreatDashboard />}
          {activeTab === 'monitoring' && <RealtimeMonitor />}
          {activeTab === 'intelligence' && <ThreatIntelligence />}
        </div>
      </main>
    </div>
  )
}
