'use client'

import { Bell } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { useState } from 'react'

export function NotificationSettings() {
  const [emailNotifications, setEmailNotifications] = useState(true)
  const [criticalAlerts, setCriticalAlerts] = useState(true)
  const [threatUpdates, setThreatUpdates] = useState(false)

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Bell className="w-5 h-5 text-primary" />
          <CardTitle>Notifications</CardTitle>
        </div>
        <CardDescription>
          Configure notification preferences
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="email">Email Notifications</Label>
            <p className="text-xs text-muted-foreground">
              Receive email alerts for important events
            </p>
          </div>
          <Switch
            id="email"
            checked={emailNotifications}
            onCheckedChange={setEmailNotifications}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="critical">Critical Alerts</Label>
            <p className="text-xs text-muted-foreground">
              Immediate notifications for critical threats
            </p>
          </div>
          <Switch
            id="critical"
            checked={criticalAlerts}
            onCheckedChange={setCriticalAlerts}
          />
        </div>

        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <Label htmlFor="threats">Threat Updates</Label>
            <p className="text-xs text-muted-foreground">
              Regular updates on threat intelligence
            </p>
          </div>
          <Switch
            id="threats"
            checked={threatUpdates}
            onCheckedChange={setThreatUpdates}
          />
        </div>
      </CardContent>
    </Card>
  )
}
