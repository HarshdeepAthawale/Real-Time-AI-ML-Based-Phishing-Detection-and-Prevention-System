'use client'

import { useState, useRef } from 'react'
import { Upload, LinkIcon, File, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { submitFileForAnalysis, submitURLForAnalysis } from '@/lib/api/sandbox'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Spinner } from '@/components/ui/loading'

interface SandboxSubmitProps {
  onSubmissionSuccess: (analysisId: string) => void
}

export function SandboxSubmit({ onSubmissionSuccess }: SandboxSubmitProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [url, setUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (file.size > 100 * 1024 * 1024) {
        setError('File size exceeds 100MB limit')
        return
      }
      setSelectedFile(file)
      setError(null)
    }
  }

  const handleFileSubmit = async () => {
    if (!selectedFile) {
      setError('Please select a file')
      return
    }

    try {
      setLoading(true)
      setError(null)
      const response = await submitFileForAnalysis(selectedFile)
      onSubmissionSuccess(response.analysis_id)
      // Reset form
      setSelectedFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    } catch (err: any) {
      setError(err.message || 'Failed to submit file for analysis')
    } finally {
      setLoading(false)
    }
  }

  const handleURLSubmit = async () => {
    if (!url.trim()) {
      setError('Please enter a URL')
      return
    }

    try {
      setLoading(true)
      setError(null)
      const response = await submitURLForAnalysis(url.trim())
      onSubmissionSuccess(response.analysis_id)
      setUrl('')
    } catch (err: any) {
      setError(err.message || 'Failed to submit URL for analysis')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Submit for Analysis</CardTitle>
        <CardDescription>
          Upload a file or submit a URL for dynamic sandbox analysis
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="file" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="file">File</TabsTrigger>
            <TabsTrigger value="url">URL</TabsTrigger>
          </TabsList>

          <TabsContent value="file" className="space-y-4">
            <div className="border-2 border-dashed border-border rounded-lg p-8 text-center">
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
                accept="*/*"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer flex flex-col items-center gap-2"
              >
                <Upload className="w-12 h-12 text-muted-foreground" />
                {selectedFile ? (
                  <div className="space-y-2">
                    <div className="flex items-center justify-center gap-2">
                      <File className="w-5 h-5 text-primary" />
                      <span className="font-medium">{selectedFile.name}</span>
                      <button
                        onClick={(e) => {
                          e.preventDefault()
                          setSelectedFile(null)
                          if (fileInputRef.current) {
                            fileInputRef.current.value = ''
                          }
                        }}
                        className="text-destructive hover:text-destructive/80"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="text-sm font-medium mb-1">
                      Click to select a file or drag and drop
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Maximum file size: 100MB
                    </p>
                  </div>
                )}
              </label>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button
              onClick={handleFileSubmit}
              disabled={loading || !selectedFile}
              className="w-full"
            >
              {loading ? (
                <>
                  <Spinner className="mr-2 h-4 w-4" />
                  Submitting...
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  Submit for Analysis
                </>
              )}
            </Button>
          </TabsContent>

          <TabsContent value="url" className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="url">URL to Analyze</Label>
              <Input
                id="url"
                type="url"
                placeholder="https://example.com"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="font-mono"
              />
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button
              onClick={handleURLSubmit}
              disabled={loading || !url.trim()}
              className="w-full"
            >
              {loading ? (
                <>
                  <Spinner className="mr-2 h-4 w-4" />
                  Submitting...
                </>
              ) : (
                <>
                  <LinkIcon className="mr-2 h-4 w-4" />
                  Submit URL for Analysis
                </>
              )}
            </Button>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

