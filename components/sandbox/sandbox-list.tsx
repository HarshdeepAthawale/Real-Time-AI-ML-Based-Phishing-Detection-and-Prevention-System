'use client'

import { useState, useEffect } from 'react'
import { File, LinkIcon, Clock, CheckCircle2, XCircle } from 'lucide-react'
import { listSandboxAnalyses, SandboxAnalysisList } from '@/lib/api/sandbox'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { TableSkeleton } from '@/components/ui/loading'
import { formatDistanceToNow } from 'date-fns'

interface SandboxListProps {
  onSelectAnalysis: (analysisId: string) => void
}

export function SandboxList({ onSelectAnalysis }: SandboxListProps) {
  const [analyses, setAnalyses] = useState<SandboxAnalysisList | null>(null)
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)

  useEffect(() => {
    const fetchAnalyses = async () => {
      try {
        setLoading(true)
        const data = await listSandboxAnalyses(page, 10)
        setAnalyses(data)
      } catch (error) {
        console.error('Error fetching analyses:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchAnalyses()
  }, [page])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-chart-1" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-destructive" />
      case 'running':
        return <Clock className="w-4 h-4 text-primary animate-pulse" />
      default:
        return <Clock className="w-4 h-4 text-muted-foreground" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-chart-1'
      case 'failed':
        return 'bg-destructive'
      case 'running':
        return 'bg-primary'
      default:
        return 'bg-muted'
    }
  }

  if (loading) {
    return <TableSkeleton rows={5} />
  }

  if (!analyses || analyses.analyses.length === 0) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-muted-foreground text-center py-8">
            No analyses found. Submit a file or URL to get started.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        {analyses.analyses.map((analysis) => (
          <Card
            key={analysis.analysis_id}
            className="cursor-pointer hover:border-primary/50 transition-colors"
            onClick={() => onSelectAnalysis(analysis.analysis_id)}
          >
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  {analysis.analysis_type === 'file' ? (
                    <File className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                  ) : (
                    <LinkIcon className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{analysis.analysis_id.slice(0, 8)}...</p>
                    <p className="text-sm text-muted-foreground">
                      {formatDistanceToNow(new Date(analysis.submitted_at), { addSuffix: true })}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={getStatusColor(analysis.status)}>
                    <span className="flex items-center gap-1">
                      {getStatusIcon(analysis.status)}
                      {analysis.status}
                    </span>
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {analyses.pagination.total_pages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Page {analyses.pagination.page} of {analyses.pagination.total_pages}
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPage((p) => p + 1)}
              disabled={page >= analyses.pagination.total_pages}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
