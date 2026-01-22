'use client'

import { useState } from 'react'
import { FlaskConical } from 'lucide-react'
import Header from '@/components/header'
import Navigation from '@/components/navigation'
import { SandboxSubmit } from '@/components/sandbox/sandbox-submit'
import { AnalysisStatus } from '@/components/sandbox/analysis-status'
import { SandboxResults } from '@/components/sandbox/sandbox-results'
import { SandboxList } from '@/components/sandbox/sandbox-list'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { SandboxAnalysis } from '@/lib/api/sandbox'

export default function SandboxPage() {
  const [selectedAnalysisId, setSelectedAnalysisId] = useState<string | null>(null)
  const [completedAnalysis, setCompletedAnalysis] = useState<SandboxAnalysis | null>(null)

  const handleSubmissionSuccess = (analysisId: string) => {
    setSelectedAnalysisId(analysisId)
  }

  const handleAnalysisComplete = (analysis: SandboxAnalysis) => {
    setCompletedAnalysis(analysis)
  }

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden">
      <Navigation activeTab="sandbox" setActiveTab={() => {}} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Header */}
            <div>
              <div className="flex items-center gap-3">
                <FlaskConical className="w-8 h-8 text-primary" />
                <div>
                  <h1 className="text-3xl font-bold text-foreground">Sandbox Analysis</h1>
                  <p className="text-muted-foreground mt-1">
                    Dynamic behavioral analysis of files and URLs
                  </p>
                </div>
              </div>
            </div>

            {/* Main Content */}
            <Tabs defaultValue="submit" className="w-full">
              <TabsList>
                <TabsTrigger value="submit">Submit</TabsTrigger>
                <TabsTrigger value="status">Status</TabsTrigger>
                <TabsTrigger value="history">History</TabsTrigger>
              </TabsList>

              <TabsContent value="submit" className="space-y-6">
                <SandboxSubmit onSubmissionSuccess={handleSubmissionSuccess} />
                {selectedAnalysisId && (
                  <AnalysisStatus
                    analysisId={selectedAnalysisId}
                    onComplete={handleAnalysisComplete}
                  />
                )}
                {completedAnalysis && <SandboxResults analysis={completedAnalysis} />}
              </TabsContent>

              <TabsContent value="status" className="space-y-6">
                {selectedAnalysisId ? (
                  <>
                    <AnalysisStatus
                      analysisId={selectedAnalysisId}
                      onComplete={handleAnalysisComplete}
                    />
                    {completedAnalysis && <SandboxResults analysis={completedAnalysis} />}
                  </>
                ) : (
                  <div className="text-center py-12">
                    <p className="text-muted-foreground">
                      No active analysis. Submit a file or URL to get started.
                    </p>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="history" className="space-y-6">
                <SandboxList onSelectAnalysis={setSelectedAnalysisId} />
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </main>
    </div>
  )
}
