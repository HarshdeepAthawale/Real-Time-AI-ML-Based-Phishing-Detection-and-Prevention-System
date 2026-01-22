'use client'

import { Search, LinkIcon, FileText, FlaskConical, Database } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useRouter } from 'next/navigation'

export function QuickActions() {
  const router = useRouter()

  const actions = [
    {
      icon: Search,
      label: 'Detect Email',
      description: 'Analyze email content',
      path: '/detection?tab=email',
      color: 'text-chart-1',
    },
    {
      icon: LinkIcon,
      label: 'Check URL',
      description: 'Analyze URL',
      path: '/detection?tab=url',
      color: 'text-chart-2',
    },
    {
      icon: FileText,
      label: 'Analyze Text',
      description: 'Check text content',
      path: '/detection?tab=text',
      color: 'text-chart-3',
    },
    {
      icon: FlaskConical,
      label: 'Sandbox Analysis',
      description: 'Submit for analysis',
      path: '/sandbox',
      color: 'text-accent',
    },
    {
      icon: Database,
      label: 'Check IOC',
      description: 'Verify indicator',
      path: '/iocs?tab=check',
      color: 'text-primary',
    },
  ]

  return (
    <Card>
      <CardContent className="pt-6">
        <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {actions.map((action) => {
            const Icon = action.icon
            return (
              <Button
                key={action.label}
                variant="outline"
                className="flex flex-col items-center gap-2 h-auto py-4 hover:bg-primary/5 hover:border-primary/50 transition-colors"
                onClick={() => router.push(action.path)}
              >
                <Icon className={`w-5 h-5 ${action.color}`} />
                <div className="text-center">
                  <p className="text-xs font-medium">{action.label}</p>
                  <p className="text-xs text-muted-foreground">{action.description}</p>
                </div>
              </Button>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
