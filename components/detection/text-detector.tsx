'use client'

import { useState } from 'react'
import { FileText, Send } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { detectText } from '@/lib/api/detection'
import { DetectionResult } from '@/lib/types/api'
import { DetectionResults } from './detection-results'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Spinner } from '@/components/ui/loading'

const textSchema = z.object({
  text: z.string().min(10, 'Text content must be at least 10 characters'),
  includeFeatures: z.boolean().optional(),
})

type TextFormData = z.infer<typeof textSchema>

export function TextDetector() {
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<TextFormData>({
    resolver: zodResolver(textSchema),
    defaultValues: {
      includeFeatures: false,
    },
  })

  const onSubmit = async (data: TextFormData) => {
    try {
      setLoading(true)
      setError(null)
      const response = await detectText({
        text: data.text,
        includeFeatures: data.includeFeatures,
      })
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'Failed to analyze text')
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
            <FileText className="w-5 h-5 text-primary" />
            <CardTitle>Text Analysis</CardTitle>
          </div>
          <CardDescription>
            Analyze text content for phishing indicators, social engineering tactics, and suspicious patterns
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="text">Text Content</Label>
              <Textarea
                id="text"
                placeholder="Paste the text content you want to analyze..."
                className="min-h-[200px]"
                {...register('text')}
              />
              {errors.text && (
                <p className="text-sm text-destructive">{errors.text.message}</p>
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
                  Analyze Text
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
