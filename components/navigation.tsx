'use client';

import { Shield, Eye, Zap, Settings, LogOut, Search, FlaskConical, Database, Rss } from 'lucide-react'
import { usePathname, useRouter } from 'next/navigation'
import Link from 'next/link'

interface NavigationProps {
  activeTab?: string
  setActiveTab?: (tab: string) => void
}

export default function Navigation({ activeTab, setActiveTab }: NavigationProps) {
  const pathname = usePathname()
  const router = useRouter()

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Shield, path: '/' },
    { id: 'detection', label: 'Detection', icon: Search, path: '/detection' },
    { id: 'monitoring', label: 'Monitoring', icon: Eye, path: '/monitoring' },
    { id: 'intelligence', label: 'Threat Intelligence', icon: Zap, path: '/intelligence' },
    { id: 'sandbox', label: 'Sandbox', icon: FlaskConical, path: '/sandbox' },
    { id: 'iocs', label: 'IOCs', icon: Database, path: '/iocs' },
    { id: 'feeds', label: 'Feeds', icon: Rss, path: '/feeds' },
  ]

  const isActive = (path: string) => {
    if (path === '/') {
      return pathname === '/'
    }
    return pathname?.startsWith(path)
  }

  const handleClick = (item: { id: string; path: string }) => {
    if (setActiveTab) {
      setActiveTab(item.id)
    }
    router.push(item.path)
  }

  return (
    <nav className="w-64 bg-sidebar border-r border-sidebar-border flex flex-col">
      {/* Logo Section */}
      <div className="p-6 border-b border-sidebar-border">
        <div className="flex items-center gap-2">
          <Shield className="w-6 h-6 text-primary" />
          <span className="font-semibold text-sidebar-foreground">Phishing Detection</span>
        </div>
      </div>

      {/* Menu Items */}
      <div className="flex-1 py-6 px-4 space-y-2 overflow-y-auto">
        {menuItems.map((item) => {
          const Icon = item.icon
          const active = isActive(item.path)
          return (
            <button
              key={item.id}
              onClick={() => handleClick(item)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                active
                  ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                  : 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="text-sm font-medium">{item.label}</span>
            </button>
          )
        })}
      </div>

      {/* Bottom Section */}
      <div className="p-4 border-t border-sidebar-border space-y-2">
        <Link
          href="/settings"
          className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
            pathname === '/settings'
              ? 'bg-sidebar-primary text-sidebar-primary-foreground'
              : 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'
          }`}
        >
          <Settings className="w-5 h-5" />
          <span className="text-sm font-medium">Settings</span>
        </Link>
        <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sidebar-foreground hover:bg-destructive hover:text-destructive-foreground transition-colors">
          <LogOut className="w-5 h-5" />
          <span className="text-sm font-medium">Logout</span>
        </button>
      </div>
    </nav>
  )
}
