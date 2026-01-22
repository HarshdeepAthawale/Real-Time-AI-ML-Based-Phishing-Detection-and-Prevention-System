'use client'

import { useState } from 'react'
import { Upload, FileText, CheckCircle2, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { bulkCheckIOC, IOCBulkCheckResponse } from '@/lib/api/ioc'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Spinner } from '@/components/ui/loading'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

export function IOCBulkChecker() {
  const [iocList, setIocList] = useState('')
  const [result, setResult] = useState<IOCBulkCheckResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const parseIOCs = (text: string) => {
    const lines = text.split('\n').filter((line) => line.trim())
    return lines.map((line) => {
      const trimmed = line.trim()
      // Try to detect type
      if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
        return { iocType: 'url' as const, iocValue: trimmed }
      }
      if (trimmed.includes('@')) {
        return { iocType: 'email' as const, iocValue: trimmed }
      }
      if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(trimmed)) {
        return { iocType: 'ip' as const, iocValue: trimmed }
      }
      if (/^[a-f0-9]{32}$/i.test(trimmed)) {
        return { iocType: 'hash_md5' as const, iocValue: trimmed }
      }
      if (/^[a-f0-9]{40}$/i.test(trimmed)) {
        return { iocType: 'hash_sha1' as const, iocValue: trimmed }
      }
      if (/^[a-f0-9]{64}$/i.test(trimmed)) {
        return { iocType: 'hash_sha256' as const, iocValue: trimmed }
      }
      // Default to domain
      return { iocType: 'domain' as const, iocValue: trimmed }
    })
  }

  const handleSubmit = async () => {
    if (!iocList.trim()) {
      setError('Please enter at least one IOC')
      return
    }

    const iocs = parseIOCs(iocList)
    if (iocs.length === 0) {
      setError('No valid IOCs found')
      return
    }

    if (iocs.length > 100) {
      setError('Maximum 100 IOCs allowed per request')
      return
    }

    try {
      setLoading(true)
      setError(null)
      const response = await bulkCheckIOC({ iocs })
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'Failed to check IOCs')
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
            <Upload className="w-5 h-5 text-primary" />
            <CardTitle>Bulk IOC Check</CardTitle>
          </div>
          <CardDescription>
            Check multiple IOCs at once (one per line, maximum 100)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="iocList">IOC List</Label>
              <Textarea
                id="iocList"
                placeholder="Enter IOCs one per line&#10;example.com&#10;192.168.1.1&#10;https://suspicious.com"
                className="min-h-[200px] font-mono text-sm"
                value={iocList}
                onChange={(e) => setIocList(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                {iocList.split('\n').filter((l) => l.trim()).length} IOCs detected
              </p>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button onClick={handleSubmit} disabled={loading || !iocList.trim()} className="w-full">
              {loading ? (
                <>
                  <Spinner className="mr-2 h-4 w-4" />
                  Checking...
                </>
              ) : (
                <>
                  <FileText className="mr-2 h-4 w-4" />
                  Check All IOCs
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Bulk Check Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="p-4 bg-muted/50 rounded-lg text-center">
                <p className="text-2xl font-bold">{result.summary.total}</p>
                <p className="text-sm text-muted-foreground">Total</p>
              </div>
              <div className="p-4 bg-destructive/10 rounded-lg text-center">
                <p className="text-2xl font-bold text-destructive">{result.summary.found}</p>
                <p className="text-sm text-muted-foreground">Found</p>
              </div>
              <div className="p-4 bg-chart-1/10 rounded-lg text-center">
                <p className="text-2xl font-bold text-chart-1">{result.summary.notFound}</p>
                <p className="text-sm text-muted-foreground">Not Found</p>
              </div>
            </div>

            <div className="border rounded-lg overflow-hidden">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>IOC Value</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Severity</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.results.map((item, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-mono text-sm">{item.ioc?.iocValue || 'N/A'}</TableCell>
                      <TableCell className="capitalize">{item.ioc?.iocType || 'N/A'}</TableCell>
                      <TableCell>
                        {item.found ? (
                          <Badge variant="destructive" className="flex items-center gap-1 w-fit">
                            <AlertTriangle className="w-3 h-3" />
                            Found
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="flex items-center gap-1 w-fit">
                            <CheckCircle2 className="w-3 h-3" />
                            Not Found
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        {item.ioc?.severity ? (
                          <Badge variant="outline">{item.ioc.severity}</Badge>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
