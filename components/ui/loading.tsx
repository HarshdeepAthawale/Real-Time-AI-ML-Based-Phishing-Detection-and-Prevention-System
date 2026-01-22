import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

interface LoadingSkeletonProps {
  className?: string
}

export function StatCardSkeleton({ className }: LoadingSkeletonProps) {
  return (
    <div className={cn('bg-card rounded-lg border border-border p-6', className)}>
      <Skeleton className="h-4 w-24 mb-2" />
      <Skeleton className="h-8 w-16 mb-1" />
      <Skeleton className="h-3 w-32" />
    </div>
  )
}

export function ThreatRowSkeleton({ className }: LoadingSkeletonProps) {
  return (
    <tr className={cn('border-b border-border', className)}>
      <td className="px-4 py-4">
        <Skeleton className="h-4 w-32" />
      </td>
      <td className="px-4 py-4">
        <Skeleton className="h-4 w-48" />
      </td>
      <td className="px-4 py-4">
        <Skeleton className="h-6 w-20 rounded-full" />
      </td>
      <td className="px-4 py-4">
        <Skeleton className="h-4 w-24" />
      </td>
      <td className="px-4 py-4">
        <Skeleton className="h-4 w-20" />
      </td>
    </tr>
  )
}

export function ChartSkeleton({ className }: LoadingSkeletonProps) {
  return (
    <div className={cn('bg-card rounded-lg border border-border p-6', className)}>
      <div className="mb-6">
        <Skeleton className="h-6 w-48 mb-2" />
        <Skeleton className="h-4 w-32" />
      </div>
      <Skeleton className="h-64 w-full" />
    </div>
  )
}

export function TableSkeleton({ rows = 5, className }: LoadingSkeletonProps & { rows?: number }) {
  return (
    <div className={cn('space-y-3', className)}>
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-16 w-full rounded-lg" />
      ))}
    </div>
  )
}

export function Spinner({ className }: LoadingSkeletonProps) {
  return (
    <div className={cn('flex items-center justify-center', className)}>
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
    </div>
  )
}
