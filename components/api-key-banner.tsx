'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Key, X } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'

const BANNER_DISMISSED_KEY = 'api_key_banner_dismissed'

export function ApiKeyBanner() {
  const [show, setShow] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || typeof window === 'undefined') return
    const apiKey = localStorage.getItem('api_key')
    const dismissed = localStorage.getItem(BANNER_DISMISSED_KEY)
    if (!apiKey?.trim() && dismissed !== 'true') {
      setShow(true)
    } else {
      setShow(false)
    }
  }, [mounted])

  const handleDismiss = () => {
    localStorage.setItem(BANNER_DISMISSED_KEY, 'true')
    setShow(false)
  }

  if (!show) return null

  return (
    <Alert className="relative border-accent/50 bg-accent/5 mb-4">
      <Key className="w-4 h-4 text-accent" />
      <AlertDescription className="pr-10">
        Configure your API key in{' '}
        <Link href="/settings" className="underline font-medium hover:text-accent">
          Settings
        </Link>
        {' '}to enable detection and real-time monitoring.
      </AlertDescription>
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-2 right-2 h-6 w-6"
        onClick={handleDismiss}
        aria-label="Dismiss"
      >
        <X className="w-4 h-4" />
      </Button>
    </Alert>
  )
}
