import { useEffect, useRef, useState } from 'react'
import * as nifti from 'nifti-reader-js'

interface Props { filePath: string }

function generateGrayColormap(): [number, number, number][] {
  const c: [number, number, number][] = []
  for (let i = 0; i < 256; i++) c.push([i, i, i])
  return c
}

function generateTurboColormap(): [number, number, number][] {
  const c: [number, number, number][] = []
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    const r = Math.min(255, Math.max(0, Math.round(255 * (0.135 + 2.43 * t - 3.82 * t * t + 2.26 * t * t * t))))
    const g = Math.min(255, Math.max(0, Math.round(255 * (0.088 + 1.85 * t - 2.84 * t * t + 1.30 * t * t * t))))
    const b = Math.min(255, Math.max(0, Math.round(255 * (0.55 - 1.20 * t + 0.60 * t * t))))
    c.push([r, g, b])
  }
  return c
}

const COLORMAPS = { gray: generateGrayColormap(), turbo: generateTurboColormap() }

export default function NiftiViewer({ filePath }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [sliceAxis, setSliceAxis] = useState(2)
  const [sliceIdx, setSliceIdx] = useState(0)
  const [colormap, setColormap] = useState<'gray' | 'turbo'>('gray')
  const dataRef = useRef<{ header: any; image: any } | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const base = await window.ctpamap.apiBase()
        const res = await fetch(`${base}/api/nifti?path=${encodeURIComponent(filePath)}`)
        const buf = await res.arrayBuffer()
        if (cancelled) return

        let dataBuf = buf
        if (nifti.isCompressed(buf)) {
          dataBuf = nifti.decompress(buf)
        }
        if (!nifti.isNIFTI(dataBuf)) {
          console.error('Not a NIfTI file')
          return
        }
        const header = nifti.readHeader(dataBuf)
        const image = nifti.readImage(header, dataBuf)
        dataRef.current = { header, image }
        setSliceIdx(Math.floor(header.dims[sliceAxis] / 2))
      } catch (e) {
        console.error('Failed to load NIfTI:', e)
      }
    }
    load()
    return () => { cancelled = true }
  }, [filePath])

  useEffect(() => {
    const d = dataRef.current
    if (!d || !canvasRef.current) return
    const { header, image } = d
    const dims = header.dims
    const w = dims[sliceAxis === 0 ? 1 : 0]
    const h = dims[sliceAxis === 2 ? 1 : 2]
    const canvas = canvasRef.current
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')!
    const imgData = ctx.createImageData(w, h)

    const data = new Float32Array(image)
    let min = Infinity, max = -Infinity
    for (let i = 0; i < data.length; i++) {
      if (data[i] !== 0) { min = Math.min(min, data[i]); max = Math.max(max, data[i]) }
    }
    if (max <= min) max = min + 1

    const cm = COLORMAPS[colormap]
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let val = 0
        if (sliceAxis === 2) {
          val = data[sliceIdx * dims[1] * dims[0] + y * dims[0] + x]
        } else if (sliceAxis === 1) {
          val = data[x * dims[1] * dims[0] + sliceIdx * dims[0] + y]
        } else {
          val = data[x * dims[1] * dims[0] + y * dims[0] + sliceIdx]
        }
        const norm = Math.max(0, Math.min(255, Math.round(((val - min) / (max - min)) * 255)))
        const c = cm[norm]
        const idx = (y * w + x) * 4
        imgData.data[idx] = c[0]
        imgData.data[idx + 1] = c[1]
        imgData.data[idx + 2] = c[2]
        imgData.data[idx + 3] = 255
      }
    }
    ctx.putImageData(imgData, 0, 0)
  }, [sliceIdx, sliceAxis, colormap])

  const maxSlice = dataRef.current ? dataRef.current.header.dims[sliceAxis] - 1 : 100

  return (
    <div>
      <div className="slider-row">
        <label>Axis:</label>
        <select value={sliceAxis} onChange={e => { setSliceAxis(+e.target.value); setSliceIdx(50) }}>
          <option value={0}>Sagittal</option>
          <option value={1}>Coronal</option>
          <option value={2}>Axial</option>
        </select>
        <label>Colormap:</label>
        <select value={colormap} onChange={e => setColormap(e.target.value as any)}>
          <option value="gray">Gray</option>
          <option value="turbo">Turbo</option>
        </select>
      </div>
      <div className="slider-row">
        <span>Slice {sliceIdx}/{maxSlice}</span>
        <input type="range" min={0} max={maxSlice} value={sliceIdx} onChange={e => setSliceIdx(+e.target.value)} />
      </div>
      <canvas ref={canvasRef} style={{ imageRendering: 'pixelated' }} />
    </div>
  )
}
