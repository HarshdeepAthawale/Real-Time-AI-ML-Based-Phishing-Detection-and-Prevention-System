'use client'

import { Palette } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ThemeToggle } from '@/components/ui/theme-toggle'
import { useTheme } from 'next-themes'

export function ThemeSettings() {
  const { theme } = useTheme()

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Palette className="w-5 h-5 text-primary" />
          <CardTitle>Theme</CardTitle>
        </div>
        <CardDescription>
          Customize the appearance of the application
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-medium">Theme Mode</p>
            <p className="text-sm text-muted-foreground">
              Current: {theme === 'system' ? 'System' : theme === 'dark' ? 'Dark' : 'Light'}
            </p>
          </div>
          <ThemeToggle />
        </div>
        <p className="text-xs text-muted-foreground">
          The theme preference is saved automatically and will persist across sessions.
        </p>
      </CardContent>
    </Card>
  )
}
