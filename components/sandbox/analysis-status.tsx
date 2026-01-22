'use client'

import { useEffect, useState } from 'react'
import { Clock, CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import { getSandboxAnalysis, SandboxAnalysis } from '@/lib/api/sandbox'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'

interface AnalysisStatusProps {
  analysisId: string
  onComplete?: (analysis: SandboxAnalysis) => void
}

export function AnalysisStatus({ analysisId, onComplete }: AnalysisStatusProps) {
  const [analysis, setAnalysis] = useState<SandboxAnalysis | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const pollAnalysis = async () => {
      try {
        const data = await getSandboxAnalysis(analysisId)
        setAnalysis(data)
        setLoading(false)

        if (data.status === 'completed' && onComplete) {
          onComplete(data)
        } else if (data.status === 'pending' || data.status === 'running') {
          // Poll again after 5 seconds
          setTimeout(pollAnalysis, 5000)
        }
      } catch (error) {
        console.error('Error polling analysis:', error)
        setLoading(false)
      }
    }

    pollAnalysis()
    const interval = setInterval(pollAnalysis, 5000)

    return () => clearInterval(interval)
  }, [analysisId, onComplete])

  if (loading && !analysis) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin text-primary" />
            <span className="text-sm text-muted-foreground">Loading analysis status...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!analysis) {
    return null
  }

  const getStatusIcon = () => {
    switch (analysis.status) {
      case 'completed':
        return <CheckCircle2 className="w-5 h-5 text-chart-1" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-destructive" />
      case 'running':
        return <Loader2 className="w-5 h-5 animate-spin text-primary" />
      default:
        return <Clock className="w-5 h-5 text-muted-foreground" />
    }
  }

  const getStatusColor = () => {
    switch (analysis.status) {
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

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <CardTitle>Analysis Status</CardTitle>
              <CardDescription>Analysis ID: {analysisId.slice(0, 8)}...</CardDescription>
            </div>
          </div>
          <Badge className={getStatusColor()}>
            {analysis.status.toUpperCase()}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-muted-foreground">Type:</span>
            <span className="ml-2 font-medium capitalize">{analysis.analysis_type}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Provider:</span>
            <span className="ml-2 font-medium">{analysis.sandbox_provider || 'N/A'}</span>
          </div>
          <div>
            <span className="text-muted-foreground">Submitted:</span>
            <span className="ml-2 font-medium">
              {new Date(analysis.submitted_at).toLocaleString()}
            </span>
          </div>
          {analysis.completed_at && (
            <div>
              <span className="text-muted-foreground">Completed:</span>
              <span className="ml-2 font-medium">
                {new Date(analysis.completed_at).toLocaleString()}
              </span>
            </div>
          )}
        </div>

        {analysis.status === 'running' && (
          <div>
            <Progress value={50} className="h-2" />
            <p className="text-xs text-muted-foreground mt-1">
              Analysis in progress...
            </p>
          </div>
        )}

        {analysis.threat && (
          <div className="pt-4 border-t border-border">
            <p className="text-sm font-semibold mb-2">Threat Detected</p>
            <div className="space-y-1 text-sm">
              <div>
                <span className="text-muted-foreground">Type:</span>
                <span className="ml-2 font-medium">{analysis.threat.threat_type}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Severity:</span>
                <Badge className="ml-2" variant="outline">
                  {analysis.threat.severity}
                </Badge>
              </div>
              <div>
                <span className="text-muted-foreground">Confidence:</span>
                <span className="ml-2 font-medium">
                  {(analysis.threat.confidence_score * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
