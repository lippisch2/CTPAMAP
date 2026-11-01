declare global {
  interface Window {
    ctpamap: {
      openFileDialog: (options?: any) => Promise<string | null>
      openDirectoryDialog: () => Promise<string | null>
      apiBase: () => Promise<string>
    }
  }
}

export interface SessionData {
  session_id: string
  meta: {
    pre_shape: number[]
    ctpa_shape: number[]
    pre_spacing: number[]
    ctpa_spacing: number[]
  }
}

export interface JobStatus {
  status: 'queued' | 'running' | 'done' | 'error'
  progress_pct: number
  stage?: string
  outputs?: Record<string, string>
  error?: string
}

export interface RunParams {
  spacing: [number, number, number]
  skip_reg: boolean
  aggressive: boolean
  lite: boolean
  method: 'default' | 'vessel' | 'antspy' | 'stix'
  hd: boolean
  no_seg: boolean
  do_aer: boolean
  use_gpu: boolean
  no_clip: boolean
}
