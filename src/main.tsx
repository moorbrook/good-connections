import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { VisionAtlas } from './vision/Atlas'
import { ResponsiveVisionBoard } from './vision/ResponsiveBoard'
import './index.css'

const rootEl = document.getElementById('root')
if (!rootEl) throw new Error('root element missing')

const root = createRoot(rootEl)
const parameters = new URLSearchParams(window.location.search)
if (parameters.has('vision-atlas')) {
  root.render(
    <StrictMode>
      <VisionAtlas />
    </StrictMode>,
  )
} else if (parameters.has('vision-board')) {
  root.render(
    <StrictMode>
      <ResponsiveVisionBoard />
    </StrictMode>,
  )
} else {
  void import('./App').then(({ App }) => {
    root.render(
      <StrictMode>
        <App />
      </StrictMode>,
    )
  })
}

// Register the hand-rolled service worker in production builds only. In dev it
// would fight Vite's HMR and serve stale modules.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    void navigator.serviceWorker.register('/sw.js')
  })
}
