'use client'

import { useState } from 'react'
import { LinkIcon, Send, ExternalLink } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { detectURL } from '@/lib/api/detection'
import { DetectionResult } from '@/lib/types/api'
import { DetectionResults } from './detection-results'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Spinner } from '@/components/ui/loading'

const urlSchema = z.object({
  url: z.string().url('Please enter a valid URL'),
  legitimateDomain: z.string().optional(),
  legitimateUrl: z.string().url('Please enter a valid URL').optional().or(z.literal('')),
})

type URLFormData = z.infer<typeof urlSchema>

export function URLDetector() {
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<URLFormData>({
    resolver: zodResolver(urlSchema),
  })

  const urlValue = watch('url')

  const onSubmit = async (data: URLFormData) => {
    try {
      setLoading(true)
      setError(null)
      const response = await detectURL({
        url: data.url,
        legitimateDomain: data.legitimateDomain || undefined,
        legitimateUrl: data.legitimateUrl || undefined,
      })
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze URL')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <LinkIcon className="w-5 h-5 text-primary" />
            <CardTitle>URL Analysis</CardTitle>
          </div>
          <CardDescription>
            Analyze URLs for phishing attempts, domain spoofing, and suspicious patterns
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="url">URL to Analyze</Label>
              <Input
                id="url"
                type="url"
                placeholder="https://example.com"
                className="font-mono"
                {...register('url')}
              />
              {errors.url && (
                <p className="text-sm text-destructive">{errors.url.message}</p>
              )}
              {urlValue && (
                <a
                  href={urlValue}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-primary hover:underline flex items-center gap-1"
                >
                  <ExternalLink className="w-3 h-3" />
                  Preview URL
                </a>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="legitimateDomain">Legitimate Domain (Optional)</Label>
                <Input
                  id="legitimateDomain"
                  placeholder="example.com"
                  {...register('legitimateDomain')}
                />
                <p className="text-xs text-muted-foreground">
                  If this URL should match a known legitimate domain
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="legitimateUrl">Legitimate URL (Optional)</Label>
                <Input
                  id="legitimateUrl"
                  type="url"
                  placeholder="https://legitimate-example.com"
                  {...register('legitimateUrl')}
                />
                <p className="text-xs text-muted-foreground">
                  Reference URL for comparison
                </p>
              </div>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button type="submit" disabled={loading} className="w-full">
              {loading ? (
                <>
                  <Spinner className="mr-2 h-4 w-4" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Send className="mr-2 h-4 w-4" />
                  Analyze URL
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {result && <DetectionResults result={result} loading={loading} />}
    </div>
  )
}
