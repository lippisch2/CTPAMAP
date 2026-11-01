import { useState } from 'react'
import { getNiftiUrl } from '../api/client'
import NiftiViewer from './NiftiViewer'
import type { JobStatus } from '../types'

interface Props { job: JobStatus | null; onBack: () => void; onNew: () => void }

export default function ResultsPanel({ job, onBack, onNew }: Props) {
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const outputs = job?.outputs || {}
  return (
    <div>
      <h2>Results</h2>
      {job?.status === 'error' && <div className="error-box">{job.error}</div>}
      <ul className="output-list">
        {Object.entries(outputs).map(([name, path]) => (
          <li key={name}>
            <span className="name">{name}</span>
            <span className="actions">
              <button className="btn-secondary" style={{ marginRight: 8 }} onClick={() => setSelectedFile(path)}>View</button>
              <a className="btn-secondary" href={getNiftiUrl(path)} download style={{ textDecoration: 'none' }}>Download</a>
            </span>
          </li>
        ))}
      </ul>
      {selectedFile && (
        <div className="viewer-container">
          <NiftiViewer filePath={selectedFile} />
        </div>
      )}
      <div className="actions">
        <button className="btn-secondary" onClick={onNew}>New Analysis</button>
      </div>
    </div>
  )
}
