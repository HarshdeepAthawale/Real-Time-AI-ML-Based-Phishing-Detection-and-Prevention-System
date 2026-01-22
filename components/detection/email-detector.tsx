'use client'

import { useState } from 'react'
import { Mail, Send } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { detectEmail } from '@/lib/api/detection'
import { DetectionResult } from '@/lib/types/api'
import { DetectionResults } from './detection-results'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Spinner } from '@/components/ui/loading'

const emailSchema = z.object({
  emailContent: z.string().min(10, 'Email content must be at least 10 characters'),
  includeFeatures: z.boolean().optional(),
})

type EmailFormData = z.infer<typeof emailSchema>

export function EmailDetector() {
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<EmailFormData>({
    resolver: zodResolver(emailSchema),
    defaultValues: {
      includeFeatures: false,
    },
  })

  const onSubmit = async (data: EmailFormData) => {
    try {
      setLoading(true)
      setError(null)
      const response = await detectEmail({
        emailContent: data.emailContent,
        includeFeatures: data.includeFeatures,
      })
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze email')
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
            <Mail className="w-5 h-5 text-primary" />
            <CardTitle>Email Analysis</CardTitle>
          </div>
          <CardDescription>
            Analyze email content for phishing attempts and social engineering indicators
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="emailContent">Email Content</Label>
              <Textarea
                id="emailContent"
                placeholder="Paste the full email content here, including headers if available..."
                className="min-h-[200px] font-mono text-sm"
                {...register('emailContent')}
              />
              {errors.emailContent && (
                <p className="text-sm text-destructive">{errors.emailContent.message}</p>
              )}
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
                  Analyze Email
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
