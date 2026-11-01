import { useState, useEffect } from 'react'
import { initApiBase, loadSession, runPipeline, getJobStatus } from './api/client'
import { ErrorBoundary } from './components/ErrorBoundary'
import InputStep from './components/InputStep'
import OptionsStep from './components/OptionsStep'
import RunStep from './components/RunStep'
import ResultsPanel from './components/ResultsPanel'
import type { SessionData, JobStatus, RunParams } from './types'

const STEPS = ['Inputs', 'Options', 'Run', 'Results']

const defaultParams: RunParams = {
  spacing: [1.5, 1.5, 1.5],
  skip_reg: false,
  aggressive: false,
  lite: false,
  method: 'default',
  hd: false,
  no_seg: false,
  do_aer: false,
  use_gpu: false,
  no_clip: false
}

function App() {
  const [step, setStep] = useState(1)
  const [prePath, setPrePath] = useState('')
  const [ctpaPath, setCtpaPath] = useState('')
  const [outDir, setOutDir] = useState('')
  const [session, setSession] = useState<SessionData | null>(null)
  const [params, setParams] = useState<RunParams>(defaultParams)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<JobStatus | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => { initApiBase() }, [])

  useEffect(() => {
    if (!jobId || job?.status === 'done' || job?.status === 'error') return
    const timer = setInterval(async () => {
      const s = await getJobStatus(jobId)
      setJob(s)
      if (s.status === 'done' || s.status === 'error') clearInterval(timer)
    }, 500)
    return () => clearInterval(timer)
  }, [jobId, job?.status])

  const handleLoad = async () => {
    setLoading(true)
    try {
      const res = await loadSession(prePath, ctpaPath, outDir || undefined)
      setSession(res)
      setStep(2)
    } catch (e: any) {
      alert(`Load failed: ${e.message}`)
    }
    setLoading(false)
  }

  const handleRun = async () => {
    if (!session) return
    setLoading(true)
    try {
      const res = await runPipeline(session.session_id, params)
      setJobId(res.job_id)
      setJob({ status: 'queued', progress_pct: 0 })
      setStep(3)
    } catch (e: any) {
      alert(`Run failed: ${e.message}`)
    }
    setLoading(false)
  }

  return (
    <ErrorBoundary>
      <div className="app">
        <header className="app-header">
          <h1>CTPAMAP</h1>
          <nav className="stepper">
            {STEPS.map((s, i) => (
              <div key={s} className={`step ${step === i + 1 ? 'active' : ''} ${step > i + 1 ? 'done' : ''}`}>
                <span className="step-num">{i + 1}</span> {s}
              </div>
            ))}
          </nav>
        </header>
        <main className="app-main">
          {step === 1 && (
            <InputStep
              prePath={prePath} setPrePath={setPrePath}
              ctpaPath={ctpaPath} setCtpaPath={setCtpaPath}
              outDir={outDir} setOutDir={setOutDir}
              onNext={handleLoad} loading={loading}
            />
          )}
          {step === 2 && (
            <OptionsStep
              params={params} setParams={setParams}
              session={session}
              onNext={handleRun} onBack={() => setStep(1)} loading={loading}
            />
          )}
          {step === 3 && (
            <RunStep job={job} onComplete={() => setStep(4)} />
          )}
          {step === 4 && (
            <ResultsPanel job={job} onBack={() => setStep(3)} onNew={() => { setStep(1); setJob(null); setJobId(null); setSession(null) }} />
          )}
        </main>
      </div>
    </ErrorBoundary>
  )
}

export default App
