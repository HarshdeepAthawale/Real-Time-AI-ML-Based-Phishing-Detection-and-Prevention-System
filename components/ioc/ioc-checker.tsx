'use client'

import { useState } from 'react'
import { Search, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { checkIOC, IOCCheckResponse } from '@/lib/api/ioc'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Spinner } from '@/components/ui/loading'

const iocCheckSchema = z.object({
  iocType: z.enum(['url', 'domain', 'ip', 'hash_md5', 'hash_sha1', 'hash_sha256', 'filename', 'email']),
  iocValue: z.string().min(1, 'IOC value is required'),
  enrich: z.boolean().optional(),
})

type IOCCheckFormData = z.infer<typeof iocCheckSchema>

export function IOCChecker() {
  const [result, setResult] = useState<IOCCheckResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue,
  } = useForm<IOCCheckFormData>({
    resolver: zodResolver(iocCheckSchema),
    defaultValues: {
      iocType: 'domain',
      enrich: false,
    },
  })

  const iocType = watch('iocType')

  const onSubmit = async (data: IOCCheckFormData) => {
    try {
      setLoading(true)
      setError(null)
      const response = await checkIOC(
        {
          iocType: data.iocType,
          iocValue: data.iocValue,
        },
        data.enrich || false
      )
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'Failed to check IOC')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const getPlaceholder = () => {
    switch (iocType) {
      case 'domain':
        return 'example.com'
      case 'url':
        return 'https://example.com'
      case 'ip':
        return '192.168.1.1'
      case 'email':
        return 'user@example.com'
      case 'hash_md5':
        return '5d41402abc4b2a76b9719d911017c592'
      case 'hash_sha1':
        return 'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d'
      case 'hash_sha256':
        return 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
      case 'filename':
        return 'suspicious.exe'
      default:
        return 'Enter IOC value'
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Search className="w-5 h-5 text-primary" />
            <CardTitle>Check IOC</CardTitle>
          </div>
          <CardDescription>
            Check if an Indicator of Compromise is known in our threat intelligence database
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="iocType">IOC Type</Label>
                <Select
                  value={iocType}
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
                <Label htmlFor="iocValue">IOC Value</Label>
                <Input
                  id="iocValue"
                  placeholder={getPlaceholder()}
                  className="font-mono"
                  {...register('iocValue')}
                />
                {errors.iocValue && (
                  <p className="text-sm text-destructive">{errors.iocValue.message}</p>
                )}
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
                  Checking...
                </>
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" />
                  Check IOC
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Check Result</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert className={result.found ? 'border-destructive' : 'border-chart-1'}>
              <div className="flex items-start gap-3">
                {result.found ? (
                  <AlertTriangle className="w-5 h-5 text-destructive" />
                ) : (
                  <CheckCircle2 className="w-5 h-5 text-chart-1" />
                )}
                <div className="flex-1">
                  <AlertDescription className="font-semibold">
                    {result.found ? 'IOC Found in Database' : 'IOC Not Found'}
                  </AlertDescription>
                  <AlertDescription className="mt-1">
                    {result.found
                      ? 'This indicator has been flagged as malicious in our threat intelligence database.'
                      : 'This indicator is not currently in our database.'}
                  </AlertDescription>
                </div>
              </div>
            </Alert>

            {result.found && result.ioc && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Value:</span>
                    <p className="font-mono font-medium mt-1">{result.ioc.iocValue}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Type:</span>
                    <p className="font-medium mt-1 capitalize">{result.ioc.iocType}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Severity:</span>
                    <Badge className="mt-1" variant="outline">
                      {result.ioc.severity}
                    </Badge>
                  </div>
                  {result.ioc.confidence && (
                    <div>
                      <span className="text-muted-foreground">Confidence:</span>
                      <p className="font-medium mt-1">{result.ioc.confidence}%</p>
                    </div>
                  )}
                </div>

                {result.ioc.threatType && (
                  <div>
                    <span className="text-muted-foreground text-sm">Threat Type:</span>
                    <p className="font-medium mt-1">{result.ioc.threatType}</p>
                  </div>
                )}

                {result.enrichment && (
                  <div className="pt-4 border-t border-border">
                    <h4 className="font-semibold mb-2">Enrichment Data</h4>
                    <pre className="p-3 bg-muted/50 rounded-lg text-xs overflow-auto">
                      {JSON.stringify(result.enrichment, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
