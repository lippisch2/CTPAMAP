import type { JobStatus } from '../types'

interface Props { job: JobStatus | null; onComplete: () => void }

export default function RunStep({ job, onComplete }: Props) {
  if (job?.status === 'done') {
    onComplete()
    return null
  }
  return (
    <div>
      <h2>Running Pipeline</h2>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${job?.progress_pct || 0}%` }}>
          {job?.progress_pct || 0}%
        </div>
      </div>
      <p style={{ textAlign: 'center', color: '#aaa' }}>
        {job?.status === 'queued' ? 'Waiting to start...' : `Stage: ${job?.stage || 'processing'}...`}
      </p>
      {job?.status === 'error' && (
        <div className="error-box" style={{ marginTop: 16 }}>
          <strong>Error:</strong> {job.error}
        </div>
      )}
    </div>
  )
}
