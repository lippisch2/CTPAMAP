import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('ctpamap', {
  openFileDialog: (options?: any) => ipcRenderer.invoke('dialog:openFile', options),
  openDirectoryDialog: () => ipcRenderer.invoke('dialog:openDirectory'),
  apiBase: () => ipcRenderer.invoke('api:base')
})
