interface Props {
  prePath: string; setPrePath: (v: string) => void
  ctpaPath: string; setCtpaPath: (v: string) => void
  outDir: string; setOutDir: (v: string) => void
  onNext: () => void; loading: boolean
}

export default function InputStep({ prePath, setPrePath, ctpaPath, setCtpaPath, outDir, setOutDir, onNext, loading }: Props) {
  const pickFile = async (setter: (v: string) => void) => {
    const p = await window.ctpamap.openFileDialog()
    if (p) setter(p)
  }
  const pickDir = async (setter: (v: string) => void) => {
    const p = await window.ctpamap.openDirectoryDialog()
    if (p) setter(p)
  }
  return (
    <div>
      <h2>Input Volumes</h2>
      <div className="form-group">
        <label>Pre-contrast CT (NIfTI)</label>
        <div className="file-input">
          <input type="text" value={prePath} readOnly placeholder="Select pre-contrast CT..." />
          <button className="btn-browse" onClick={() => pickFile(setPrePath)}>Browse</button>
        </div>
      </div>
      <div className="form-group">
        <label>Contrast-enhanced CTPA (NIfTI)</label>
        <div className="file-input">
          <input type="text" value={ctpaPath} readOnly placeholder="Select CTPA..." />
          <button className="btn-browse" onClick={() => pickFile(setCtpaPath)}>Browse</button>
        </div>
      </div>
      <div className="form-group">
        <label>Output Directory (optional)</label>
        <div className="file-input">
          <input type="text" value={outDir} readOnly placeholder="Default: input directory" />
          <button className="btn-browse" onClick={() => pickDir(setOutDir)}>Browse</button>
        </div>
      </div>
      <div className="actions">
        <button className="btn-primary" disabled={!prePath || !ctpaPath || loading} onClick={onNext}>
          {loading ? 'Loading...' : 'Load & Continue'}
        </button>
      </div>
    </div>
  )
}
