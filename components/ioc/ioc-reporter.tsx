'use client'

import { useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { reportIOC } from '@/lib/api/ioc'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Spinner } from '@/components/ui/loading'
import { useToast } from '@/hooks/use-toast'

const iocReportSchema = z.object({
  iocType: z.enum(['url', 'domain', 'ip', 'hash_md5', 'hash_sha1', 'hash_sha256', 'filename', 'email']),
  iocValue: z.string().min(1, 'IOC value is required'),
  threatType: z.string().optional(),
  severity: z.enum(['critical', 'high', 'medium', 'low']).optional(),
  confidence: z.number().min(0).max(100).optional(),
})

type IOCReportFormData = z.infer<typeof iocReportSchema>

export function IOCReporter() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const { toast } = useToast()

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    watch,
    setValue,
  } = useForm<IOCReportFormData>({
    resolver: zodResolver(iocReportSchema),
    defaultValues: {
      iocType: 'domain',
      severity: 'medium',
      confidence: 70,
    },
  })

  const onSubmit = async (data: IOCReportFormData) => {
    try {
      setLoading(true)
      setError(null)
      setSuccess(false)
      await reportIOC({
        iocType: data.iocType,
        iocValue: data.iocValue,
        threatType: data.threatType,
        severity: data.severity,
        confidence: data.confidence,
      })
      setSuccess(true)
      reset()
      toast({
        title: 'IOC Reported',
        description: 'The IOC has been successfully reported to the threat intelligence database.',
      })
    } catch (err: any) {
      setError(err.message || 'Failed to report IOC')
      setSuccess(false)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-primary" />
          <CardTitle>Report IOC</CardTitle>
        </div>
        <CardDescription>
          Report a new Indicator of Compromise to the threat intelligence database
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="iocType">IOC Type</Label>
              <Select
                value={watch('iocType')}
                onValueChange={(value) => setValue('iocType', value as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="domain">Domain</SelectItem>
                  <SelectItem value="url">URL</SelectItem>
                  <SelectItem value="ip">IP Address</SelectItem>
                  <SelectItem value="email">Email</SelectItem>
                  <SelectItem value="hash_md5">MD5 Hash</SelectItem>
                  <SelectItem value="hash_sha1">SHA1 Hash</SelectItem>
                  <SelectItem value="hash_sha256">SHA256 Hash</SelectItem>
                  <SelectItem value="filename">Filename</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="iocValue">IOC Value *</Label>
              <Input
                id="iocValue"
                placeholder="Enter IOC value"
                className="font-mono"
                {...register('iocValue')}
              />
              {errors.iocValue && (
                <p className="text-sm text-destructive">{errors.iocValue.message}</p>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="threatType">Threat Type (Optional)</Label>
              <Input
                id="threatType"
                placeholder="e.g., Phishing, Malware"
                {...register('threatType')}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="severity">Severity</Label>
              <Select
                value={watch('severity')}
                onValueChange={(value) => setValue('severity', value as any)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="critical">Critical</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="confidence">Confidence (%)</Label>
              <Input
                id="confidence"
                type="number"
                min="0"
                max="100"
                {...register('confidence', { valueAsNumber: true })}
              />
            </div>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {success && (
            <Alert className="border-chart-1">
              <AlertDescription>
                IOC successfully reported to the threat intelligence database.
              </AlertDescription>
            </Alert>
          )}

          <Button type="submit" disabled={loading} className="w-full">
            {loading ? (
              <>
                <Spinner className="mr-2 h-4 w-4" />
                Reporting...
              </>
            ) : (
              <>
                <AlertTriangle className="mr-2 h-4 w-4" />
                Report IOC
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
