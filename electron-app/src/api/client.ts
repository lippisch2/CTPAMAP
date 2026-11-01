let _apiBase = ''

export async function initApiBase() {
  _apiBase = await window.ctpamap.apiBase()
}

export async function apiFetch(path: string, options?: RequestInit) {
  const res = await fetch(`${_apiBase}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers }
  })
  return res.json()
}

export async function loadSession(pre: string, ctpa: string, out?: string) {
  return apiFetch('/api/load', {
    method: 'POST',
    body: JSON.stringify({ pre, ctpa, out })
  })
}

export async function runPipeline(sessionId: string, params: Record<string, any>) {
  return apiFetch('/api/run', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, params })
  })
}

export async function getJobStatus(jobId: string) {
  const res = await fetch(`${_apiBase}/api/run/${jobId}`)
  return res.json()
}

export function getNiftiUrl(path: string) {
  return `${_apiBase}/api/nifti?path=${encodeURIComponent(path)}`
}
