import { AlertTriangle, CheckCircle2, Clock, Shield } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

interface Threat {
  id: string
  type: string
  target: string
  severity: 'critical' | 'high' | 'medium'
  status: 'blocked' | 'monitored' | 'resolved'
  timestamp: string
}

export default function RecentThreats() {
  const threats: Threat[] = [
    {
      id: '1',
      type: 'Email Phishing',
      target: 'enterprise@company.com',
      severity: 'critical',
      status: 'blocked',
      timestamp: '2 minutes ago',
    },
    {
      id: '2',
      type: 'URL Spoofing',
      target: 'attacker-bank-clone.io',
      severity: 'high',
      status: 'monitored',
      timestamp: '15 minutes ago',
    },
    {
      id: '3',
      type: 'Domain Hijacking',
      target: 'secure-mail-service.co',
      severity: 'high',
      status: 'blocked',
      timestamp: '1 hour ago',
    },
    {
      id: '4',
      type: 'AI-Generated Content',
      target: 'noreply@trusted-source.fake',
      severity: 'medium',
      status: 'resolved',
      timestamp: '3 hours ago',
    },
    {
      id: '5',
      type: 'Credential Harvesting',
      target: 'login-verify.malicious.net',
      severity: 'critical',
      status: 'blocked',
      timestamp: '5 hours ago',
    },
  ]

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-destructive/20 text-destructive border-destructive/30'
      case 'high':
        return 'bg-accent/20 text-accent border-accent/30'
      case 'medium':
        return 'bg-chart-3/20 text-chart-3 border-chart-3/30'
      default:
        return 'bg-chart-2/20 text-chart-2 border-chart-2/30'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'blocked':
        return <Shield className="w-4 h-4 text-chart-1" />
      case 'monitored':
        return <Clock className="w-4 h-4 text-chart-3" />
      case 'resolved':
        return <CheckCircle2 className="w-4 h-4 text-muted-foreground" />
      default:
        return null
    }
  }

  return (
    <div className="bg-card rounded-lg border border-border p-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">Recent Threats</h3>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Type
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Target
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Severity
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Status
              </th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
                Time
              </th>
            </tr>
          </thead>
          <tbody>
            {threats.map((threat) => (
              <tr
                key={threat.id}
                className="border-b border-border hover:bg-muted/50 transition-colors"
              >
                <td className="px-4 py-4 text-sm text-foreground font-medium">{threat.type}</td>
                <td className="px-4 py-4 text-sm text-muted-foreground font-mono">{threat.target}</td>
                <td className="px-4 py-4 text-sm">
                  <Badge
                    variant="outline"
                    className={`capitalize ${getSeverityColor(threat.severity)}`}
                  >
                    {threat.severity}
                  </Badge>
                </td>
                <td className="px-4 py-4 text-sm">
                  <div className="flex items-center gap-2">
                    {getStatusIcon(threat.status)}
                    <span className="capitalize text-muted-foreground">{threat.status}</span>
                  </div>
                </td>
                <td className="px-4 py-4 text-sm text-muted-foreground">{threat.timestamp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
