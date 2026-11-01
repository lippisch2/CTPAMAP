import type { RunParams, SessionData } from '../types'

interface Props {
  params: RunParams; setParams: (p: RunParams) => void
  session: SessionData | null
  onNext: () => void; onBack: () => void; loading: boolean
}

export default function OptionsStep({ params, setParams, session, onNext, onBack, loading }: Props) {
  const update = <K extends keyof RunParams>(key: K, val: RunParams[K]) => setParams({ ...params, [key]: val })
  const meta = session?.meta
  return (
    <div>
      <h2>Registration Options</h2>
      {meta && (
        <div className="section-title">
          Pre: {meta.pre_shape.join('x')} | CTPA: {meta.ctpa_shape.join('x')}
        </div>
      )}
      <div className="section-title">Registration Method</div>
      <div className="radio-group">
        {(['default', 'vessel', 'antspy', 'stix'] as const).map(m => (
          <label key={m} className="radio-item">
            <input type="radio" checked={params.method === m} onChange={() => update('method', m)} />
            {m === 'default' ? 'Default (MI+Demons)' : m === 'vessel' ? 'Vesselness (recommended)' : m === 'antspy' ? 'ANTsPy SyN' : 'ITK-Elastix'}
          </label>
        ))}
      </div>
      <div className="section-title">Registration Control</div>
      <div className="checkbox-group">
        <label className="checkbox-item"><input type="checkbox" checked={params.skip_reg} onChange={e => update('skip_reg', e.target.checked)} /> Skip registration</label>
        <label className="checkbox-item"><input type="checkbox" checked={params.aggressive} onChange={e => update('aggressive', e.target.checked)} /> Aggressive</label>
        <label className="checkbox-item"><input type="checkbox" checked={params.lite} onChange={e => update('lite', e.target.checked)} /> Aggressive lite</label>
        <label className="checkbox-item"><input type="checkbox" checked={params.hd} onChange={e => update('hd', e.target.checked)} /> High-definition spacing</label>
      </div>
      <div className="section-title">Segmentation & Correction</div>
      <div className="checkbox-group">
        <label className="checkbox-item"><input type="checkbox" checked={params.no_seg} onChange={e => update('no_seg', e.target.checked)} /> Skip segmentation</label>
        <label className="checkbox-item"><input type="checkbox" checked={params.do_aer} onChange={e => update('do_aer', e.target.checked)} /> Aeration correction</label>
        <label className="checkbox-item"><input type="checkbox" checked={params.use_gpu} onChange={e => update('use_gpu', e.target.checked)} /> GPU (FireANTs)</label>
        <label className="checkbox-item"><input type="checkbox" checked={params.no_clip} onChange={e => update('no_clip', e.target.checked)} /> Skip HU clip</label>
      </div>
      <div className="section-title">Output</div>
      <div className="form-group">
        <label>Target spacing (mm)</label>
        <div className="form-row">
          {[0, 1, 2].map(i => (
            <input key={i} type="number" value={params.spacing[i]} step={0.1} min={0.1}
              onChange={e => { const s = [...params.spacing] as [number, number, number]; s[i] = parseFloat(e.target.value) || 1.5; update('spacing', s) }} />
          ))}
        </div>
      </div>
      <div className="actions">
        <button className="btn-secondary" onClick={onBack}>Back</button>
        <button className="btn-primary" onClick={onNext} disabled={loading}>Run Pipeline</button>
      </div>
    </div>
  )
}
