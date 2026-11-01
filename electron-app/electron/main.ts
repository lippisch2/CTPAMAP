import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import { join } from 'path'
import { spawn, ChildProcess } from 'child_process'

let mainWindow: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null
const API_BASE = 'http://127.0.0.1:18923'

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: join(__dirname, '../preload/preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    title: 'CTPAMAP',
    icon: undefined
  })

  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  mainWindow.on('closed', () => { mainWindow = null })
}

function startPythonBackend() {
  // In development the project root is three levels above dist-electron/main.
  // In a packaged build, use the directory the app was launched from, and fall
  // back to an installed `ctpamap` package in site-packages.
  const cwd = app.isPackaged
    ? process.cwd()
    : join(__dirname, '../../..')
  const pySep = process.platform === 'win32' ? ';' : ':'
  const pythonBin =
    process.env.CTPAMAP_PYTHON ||
    (process.platform === 'win32' ? 'python' : 'python3')
  const env = {
    ...process.env,
    PYTHONPATH: cwd + (process.env.PYTHONPATH ? pySep + process.env.PYTHONPATH : '')
  }
  pythonProcess = spawn(pythonBin, ['-m', 'uvicorn', 'ctpamap.api:app', '--host', '127.0.0.1', '--port', '18923'], {
    cwd,
    env,
    stdio: ['ignore', 'pipe', 'pipe']
  })
  pythonProcess.stdout?.on('data', (d) => console.log(`[python] ${d}`))
  pythonProcess.stderr?.on('data', (d) => console.log(`[python] ${d}`))
  pythonProcess.on('error', (err) => console.error('Python spawn error:', err))
}

async function waitForBackend(maxRetries = 60, intervalMs = 500): Promise<boolean> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const res = await fetch(`${API_BASE}/api/status`)
      if (res.ok) return true
    } catch {}
    await new Promise(r => setTimeout(r, intervalMs))
  }
  return false
}

app.whenReady().then(async () => {
  startPythonBackend()
  const ready = await waitForBackend()
  if (!ready) {
    console.error('Backend failed to start')
    app.quit()
    return
  }
  createWindow()
})

app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill()
    pythonProcess = null
  }
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

ipcMain.handle('dialog:openFile', async (_, options) => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ['openFile'],
    filters: [
      { name: 'NIfTI', extensions: ['nii', 'nii.gz'] },
      { name: 'All Files', extensions: ['*'] }
    ],
    ...(options || {})
  })
  return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('dialog:openDirectory', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ['openDirectory']
  })
  return result.canceled ? null : result.filePaths[0]
})

ipcMain.handle('api:base', () => API_BASE)
