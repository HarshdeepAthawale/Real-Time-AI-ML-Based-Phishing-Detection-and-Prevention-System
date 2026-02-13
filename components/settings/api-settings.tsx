'use client'

import { useState, useEffect } from 'react'
import { Key, Eye, EyeOff, Copy, Check, Globe } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { useToast } from '@/hooks/use-toast'

const DEFAULT_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'

export function ApiSettings() {
  const [apiKey, setApiKey] = useState('')
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_URL)
  const [showKey, setShowKey] = useState(false)
  const [copied, setCopied] = useState(false)
  const { toast } = useToast()

  // Load API key and URL from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const storedKey = localStorage.getItem('api_key')
      if (storedKey) setApiKey(storedKey)
      const storedUrl = localStorage.getItem('api_url')
      if (storedUrl) setApiUrl(storedUrl)
    }
  }, [])

  const handleSave = () => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('api_key', apiKey)
      localStorage.setItem('api_url', apiUrl.trim() || DEFAULT_API_URL)
      toast({
        title: 'Settings Saved',
        description: 'API URL and key have been saved. Refresh the page for URL changes to take effect.',
      })
    }
  }

  const handleCopy = () => {
    if (apiKey) {
      navigator.clipboard.writeText(apiKey)
      setCopied(true)
      toast({
        title: 'Copied',
        description: 'API key copied to clipboard',
      })
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Key className="w-5 h-5 text-primary" />
          <CardTitle>API Key</CardTitle>
        </div>
        <CardDescription>
          Configure your API key for backend authentication
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="apiUrl">API Base URL</Label>
          <div className="flex gap-2 items-center">
            <Globe className="w-4 h-4 text-muted-foreground" />
            <Input
              id="apiUrl"
              type="url"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:3000"
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Backend API gateway or detection API URL (e.g. http://localhost:3000)
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="apiKey">API Key</Label>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                id="apiKey"
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your API key"
                className="pr-10"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute right-0 top-0 h-full"
                onClick={() => setShowKey(!showKey)}
              >
                {showKey ? (
                  <EyeOff className="w-4 h-4" />
                ) : (
                  <Eye className="w-4 h-4" />
                )}
              </Button>
            </div>
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={handleCopy}
              disabled={!apiKey}
            >
              {copied ? (
                <Check className="w-4 h-4" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Your API key is stored locally in your browser
          </p>
        </div>

        <Alert>
          <AlertDescription>
            Keep your API key secure. Do not share it publicly or commit it to version control.
          </AlertDescription>
        </Alert>

        <Button onClick={handleSave} className="w-full">
          Save API Key
        </Button>
      </CardContent>
    </Card>
  )
}
