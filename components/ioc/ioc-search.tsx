'use client'

import { useState, useEffect } from 'react'
import { Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { searchIOCs, IOCSearchResponse } from '@/lib/api/ioc'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { TableSkeleton } from '@/components/ui/loading'
import { formatDistanceToNow } from 'date-fns'

export function IOCSearch() {
  const [searchTerm, setSearchTerm] = useState('')
  const [iocType, setIocType] = useState<string>('')
  const [severity, setSeverity] = useState<string>('')
  const [results, setResults] = useState<IOCSearchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(0)

  const handleSearch = async () => {
    try {
      setLoading(true)
      const response = await searchIOCs({
        search: searchTerm || undefined,
        iocType: iocType || undefined,
        severity: severity as any || undefined,
        limit: 20,
        offset: page * 20,
      })
      setResults(response)
    } catch (error) {
      console.error('Error searching IOCs:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    handleSearch()
  }, [page])

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Search className="w-5 h-5 text-primary" />
          <CardTitle>Search IOCs</CardTitle>
        </div>
        <CardDescription>
          Search the threat intelligence database for Indicators of Compromise
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor="search">Search</Label>
            <Input
              id="search"
              placeholder="Search IOCs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="iocType">Type</Label>
            <Select value={iocType} onValueChange={setIocType}>
              <SelectTrigger>
                <SelectValue placeholder="All types" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All types</SelectItem>
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
            <Label htmlFor="severity">Severity</Label>
            <Select value={severity} onValueChange={setSeverity}>
              <SelectTrigger>
                <SelectValue placeholder="All severities" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">All severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <Button onClick={handleSearch} disabled={loading} className="w-full">
          {loading ? 'Searching...' : 'Search'}
        </Button>

        {loading && <TableSkeleton rows={5} />}

        {results && !loading && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-sm text-muted-foreground">
                Found {results.total} IOCs
              </p>
            </div>

            {results.iocs.length === 0 ? (
              <p className="text-center py-8 text-muted-foreground">
                No IOCs found matching your criteria
              </p>
            ) : (
              <>
                <div className="border rounded-lg overflow-hidden">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Value</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Severity</TableHead>
                        <TableHead>Threat Type</TableHead>
                        <TableHead>First Seen</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {results.iocs.map((ioc, index) => (
                        <TableRow key={index}>
                          <TableCell className="font-mono text-sm">{ioc.iocValue}</TableCell>
                          <TableCell className="capitalize">{ioc.iocType}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{ioc.severity}</Badge>
                          </TableCell>
                          <TableCell>{ioc.threatType || '-'}</TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {ioc.firstSeenAt
                              ? formatDistanceToNow(new Date(ioc.firstSeenAt), { addSuffix: true })
                              : '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                {results.total > 20 && (
                  <div className="flex items-center justify-between">
                    <Button
                      variant="outline"
                      onClick={() => setPage((p) => Math.max(0, p - 1))}
                      disabled={page === 0}
                    >
                      Previous
                    </Button>
                    <span className="text-sm text-muted-foreground">
                      Page {page + 1} of {Math.ceil(results.total / 20)}
                    </span>
                    <Button
                      variant="outline"
                      onClick={() => setPage((p) => p + 1)}
                      disabled={(page + 1) * 20 >= results.total}
                    >
                      Next
                    </Button>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
