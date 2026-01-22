'use client'

import { AlertTriangle, CheckCircle2, XCircle, Info } from 'lucide-react'
import { DetectionResult } from '@/lib/types/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'

interface DetectionResultsProps {
  result: DetectionResult | null
  loading?: boolean
}

export function DetectionResults({ result, loading }: DetectionResultsProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Analyzing...</CardTitle>
          <CardDescription>Processing your request</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="h-4 bg-muted animate-pulse rounded" />
            <div className="h-4 bg-muted animate-pulse rounded w-3/4" />
            <div className="h-4 bg-muted animate-pulse rounded w-1/2" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!result) {
    return null
  }

  const severityColors = {
    critical: 'bg-destructive text-destructive-foreground',
    high: 'bg-orange-500 text-white',
    medium: 'bg-yellow-500 text-white',
    low: 'bg-blue-500 text-white',
  }

  const getSeverityIcon = () => {
    if (!result.isThreat) {
      return <CheckCircle2 className="w-5 h-5 text-chart-1" />
    }
    switch (result.severity) {
      case 'critical':
        return <AlertTriangle className="w-5 h-5 text-destructive" />
      case 'high':
        return <AlertTriangle className="w-5 h-5 text-orange-500" />
      default:
        return <Info className="w-5 h-5 text-yellow-500" />
    }
  }

  return (
    <div className="space-y-4">
      <Alert className={result.isThreat ? 'border-destructive' : 'border-chart-1'}>
        <div className="flex items-start gap-3">
          {getSeverityIcon()}
          <div className="flex-1">
            <AlertDescription className="font-semibold">
              {result.isThreat ? 'Threat Detected' : 'No Threat Detected'}
            </AlertDescription>
            <AlertDescription className="mt-1">
              {result.isThreat
                ? `This content has been identified as ${result.threatType} with ${(result.confidence * 100).toFixed(1)}% confidence.`
                : `This content appears to be legitimate with ${((1 - result.confidence) * 100).toFixed(1)}% confidence.`}
            </AlertDescription>
          </div>
        </div>
      </Alert>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Detection Results</CardTitle>
              <CardDescription>Detailed analysis breakdown</CardDescription>
            </div>
            <Badge className={severityColors[result.severity]}>
              {result.severity.toUpperCase()}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Confidence Score */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Overall Confidence</span>
              <span className="text-sm font-semibold">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
            <Progress value={result.confidence * 100} className="h-2" />
          </div>

          {/* ML Service Scores */}
          <div>
            <h4 className="text-sm font-semibold mb-3">ML Service Scores</h4>
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">Ensemble Score</span>
                  <span className="text-xs font-medium">{(result.scores.ensemble * 100).toFixed(1)}%</span>
                </div>
                <Progress value={result.scores.ensemble * 100} className="h-1.5" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">NLP Analysis</span>
                  <span className="text-xs font-medium">{(result.scores.nlp * 100).toFixed(1)}%</span>
                </div>
                <Progress value={result.scores.nlp * 100} className="h-1.5" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">URL Analysis</span>
                  <span className="text-xs font-medium">{(result.scores.url * 100).toFixed(1)}%</span>
                </div>
                <Progress value={result.scores.url * 100} className="h-1.5" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-muted-foreground">Visual Analysis</span>
                  <span className="text-xs font-medium">{(result.scores.visual * 100).toFixed(1)}%</span>
                </div>
                <Progress value={result.scores.visual * 100} className="h-1.5" />
              </div>
            </div>
          </div>

          {/* Threat Indicators */}
          {result.indicators && result.indicators.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-2">Threat Indicators</h4>
              <div className="flex flex-wrap gap-2">
                {result.indicators.map((indicator, index) => (
                  <Badge key={index} variant="outline" className="text-xs">
                    {indicator}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Metadata */}
          <div className="pt-4 border-t border-border">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Processing Time:</span>
                <span className="ml-2 font-medium">{result.metadata.processingTimeMs}ms</span>
              </div>
              <div>
                <span className="text-muted-foreground">Threat Type:</span>
                <span className="ml-2 font-medium">{result.threatType}</span>
              </div>
              {result.cached && (
                <div className="col-span-2">
                  <Badge variant="secondary" className="text-xs">
                    Result served from cache
                  </Badge>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
