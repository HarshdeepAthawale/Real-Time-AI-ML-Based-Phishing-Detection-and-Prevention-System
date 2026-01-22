'use client'

import { Settings as SettingsIcon } from 'lucide-react'
import Header from '@/components/header'
import Navigation from '@/components/navigation'
import { ApiSettings } from '@/components/settings/api-settings'
import { ThemeSettings } from '@/components/settings/theme-settings'
import { NotificationSettings } from '@/components/settings/notification-settings'
import { DisplaySettings } from '@/components/settings/display-settings'

export default function SettingsPage() {
  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Navigation />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Header */}
            <div>
              <div className="flex items-center gap-3">
                <SettingsIcon className="w-8 h-8 text-primary" />
                <div>
                  <h1 className="text-3xl font-bold text-foreground">Settings</h1>
                  <p className="text-muted-foreground mt-1">
                    Manage your application preferences and configuration
                  </p>
                </div>
              </div>
            </div>

            {/* Settings Sections */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ApiSettings />
              <ThemeSettings />
              <NotificationSettings />
              <DisplaySettings />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
