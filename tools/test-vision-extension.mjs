#!/usr/bin/env node

import { spawn } from 'node:child_process'
import { access, cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import { homedir, tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const projectRoot = resolve(fileURLToPath(new URL('..', import.meta.url)))
const chromeCandidates = [
  process.env.CHROME_BIN,
  join(homedir(), 'Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing'),
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean)

async function freePort() {
  return new Promise((resolvePort, reject) => {
    const server = createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      server.close(() => resolvePort(address.port))
    })
  })
}

async function waitFor(check, label, timeoutMs = 15_000) {
  const deadline = Date.now() + timeoutMs
  let lastError
  while (Date.now() < deadline) {
    try {
      const value = await check()
      if (value) return value
    } catch (error) {
      lastError = error
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 100))
  }
  throw new Error(`Timed out waiting for ${label}${lastError ? `: ${lastError}` : ''}`)
}

class Cdp {
  constructor(url) {
    this.socket = new WebSocket(url)
    this.nextId = 1
    this.pending = new Map()
  }

  async open() {
    await new Promise((resolveOpen, reject) => {
      this.socket.addEventListener('open', resolveOpen, { once: true })
      this.socket.addEventListener('error', reject, { once: true })
    })
    this.socket.addEventListener('message', ({ data }) => {
      const message = JSON.parse(data)
      if (!message.id) return
      const pending = this.pending.get(message.id)
      if (!pending) return
      this.pending.delete(message.id)
      if (message.error) pending.reject(new Error(message.error.message))
      else pending.resolve(message.result)
    })
    return this
  }

  send(method, params = {}) {
    const id = this.nextId++
    return new Promise((resolveSend, reject) => {
      this.pending.set(id, { resolve: resolveSend, reject })
      this.socket.send(JSON.stringify({ id, method, params }))
    })
  }

  async evaluate(expression) {
    const result = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    })
    if (result.exceptionDetails) {
      throw new Error(
        result.exceptionDetails.exception?.description ?? result.exceptionDetails.text,
      )
    }
    return result.result.value
  }

  close() {
    this.socket.close()
  }
}

function isSet(ids) {
  const cards = ids.map((id) => [
    Math.floor(id / 27) % 3,
    Math.floor(id / 9) % 3,
    Math.floor(id / 3) % 3,
    id % 3,
  ])
  return [0, 1, 2, 3].every((attribute) =>
    (cards[0][attribute] + cards[1][attribute] + cards[2][attribute]) % 3 === 0)
}

const processes = []
const browserDiagnostics = []
let temporaryRoot
try {
  const chrome = await waitFor(async () => {
    for (const candidate of chromeCandidates) {
      try {
        await access(candidate)
        return candidate
      } catch {}
    }
    return undefined
  }, 'a Chrome binary', 500)
  const previewPort = await freePort()
  const debugPort = await freePort()
  temporaryRoot = await mkdtemp(join(tmpdir(), 'set-vision-extension-test.'))
  const extensionDir = join(temporaryRoot, 'extension')
  const profileDir = join(temporaryRoot, 'profile')
  await cp(join(projectRoot, 'extension'), extensionDir, { recursive: true })
  const manifestPath = join(extensionDir, 'manifest.json')
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
  // Headless CDP clicks do not grant activeTab's browser-gesture capability.
  // Broaden only this throwaway manifest so captureVisibleTab can exercise the
  // same screenshot path; the production extension remains activeTab-only.
  manifest.host_permissions = ['<all_urls>']
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)

  const preview = spawn('npm', [
    'run', 'preview', '--', '--host', '127.0.0.1', '--port', String(previewPort), '--strictPort',
  ], { cwd: projectRoot, stdio: ['ignore', 'ignore', 'pipe'] })
  processes.push(preview)
  await waitFor(async () => (await fetch(`http://127.0.0.1:${previewPort}/`)).ok, 'Vite preview')

  const browser = spawn(chrome, [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    `--remote-debugging-port=${debugPort}`,
    `--user-data-dir=${profileDir}`,
    `--disable-extensions-except=${extensionDir}`,
    `--load-extension=${extensionDir}`,
    '--window-size=1600,1000',
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] })
  browser.stderr.on('data', (chunk) => browserDiagnostics.push(String(chunk)))
  processes.push(browser)

  const extensionTarget = await waitFor(async () => {
    const response = await fetch(`http://127.0.0.1:${debugPort}/json/list`)
    const list = await response.json()
    for (const target of list.filter((candidate) => candidate.url.endsWith('/background.js'))) {
      const candidate = await new Cdp(target.webSocketDebuggerUrl).open()
      await candidate.send('Runtime.enable')
      const name = await candidate.evaluate('chrome.runtime.getManifest().name')
      candidate.close()
      if (name === manifest.name) return target
    }
    return undefined
  }, 'vision extension service worker')
  const extensionId = new URL(extensionTarget.url).host
  const createTarget = async (url) => {
    const response = await fetch(
      `http://127.0.0.1:${debugPort}/json/new?${encodeURIComponent(url)}`,
      { method: 'PUT' },
    )
    return response.json()
  }
  const gameTarget = await createTarget(`http://127.0.0.1:${previewPort}/`)
  const worker = await new Cdp(extensionTarget.webSocketDebuggerUrl).open()
  await worker.send('Runtime.enable')
  const panelResource = await worker.evaluate(`fetch(chrome.runtime.getURL('sidepanel.html'))
    .then(async (response) => ({ ok: response.ok, status: response.status, text: await response.text() }))
    .catch((error) => ({ error: String(error) }))`)
  if (!panelResource?.ok) {
    const loadedManifest = await worker.evaluate('chrome.runtime.getManifest()')
    throw new Error(
      `Extension page resource is unavailable for ${extensionTarget.url} ${JSON.stringify(loadedManifest)}: ${JSON.stringify(panelResource)}`,
    )
  }
  const createdPanelTab = await worker.evaluate(`chrome.tabs.create({
    url: chrome.runtime.getURL('sidepanel.html'),
    active: false,
  })`)
  if (!createdPanelTab?.id) {
    throw new Error(`Extension page tab was not created: ${JSON.stringify(createdPanelTab)}`)
  }
  const panelTarget = await waitFor(async () => {
    const response = await fetch(`http://127.0.0.1:${debugPort}/json/list`)
    const list = await response.json()
    return list.find((target) => target.url === `chrome-extension://${extensionId}/sidepanel.html`)
  }, 'extension page')
  worker.close()
  const game = await new Cdp(gameTarget.webSocketDebuggerUrl).open()
  const panel = await new Cdp(panelTarget.webSocketDebuggerUrl).open()
  await game.send('Page.enable')
  await panel.send('Page.enable')
  try {
    await waitFor(async () => {
      const value = await panel.evaluate("document.querySelector('#status')?.textContent")
      if (value?.includes('ready')) return value
      if (value && value !== 'Loading local Rust model…') {
        throw new Error(`extension status: ${value}`)
      }
      return undefined
    }, 'FrankenTorch extension initialization')
  } catch (error) {
    const href = await panel.evaluate('location.href')
    const probe = await panel.evaluate(`(async () => {
      try {
        const module = await import('./set_vision_runtime.js')
        const runtime = await module.default({
          module_or_path: chrome.runtime.getURL('set_vision_runtime_bg.wasm'),
        })
        return { ready: runtime.model_ready() }
      } catch (failure) {
        return { error: String(failure), stack: failure?.stack }
      }
    })()`)
    const stderr = browserDiagnostics.join('').split('\n').slice(-20).join('\n')
    throw new Error(
      `${error}; created tab: ${JSON.stringify(createdPanelTab)}; panel URL: ${href}; direct runtime probe: ${JSON.stringify(probe)}; browser stderr: ${stderr}`,
    )
  }

  const cases = []
  for (const viewport of [[390, 800], [600, 900], [900, 1000]]) {
    for (const theme of ['classic', 'nocturne', 'atelier', 'signal']) {
      cases.push({ viewport, theme })
    }
  }

  let modelFailures = 0
  let actionFailures = 0
  let acceptedActions = 0
  for (const [index, testCase] of cases.entries()) {
    const [width, height] = testCase.viewport
    await game.send('Emulation.setDeviceMetricsOverride', {
      width, height, deviceScaleFactor: 1, mobile: false,
    })
    await game.send('Page.navigate', {
      url: `http://127.0.0.1:${previewPort}/?vision-board=${testCase.theme}&board-seed=${100 + index}&vision-seed=${9000 + index}`,
    })
    await waitFor(
      () => game.evaluate("document.documentElement.dataset.visionBoardReady === 'true'"),
      'responsive board',
    )
    const cardRect = await game.evaluate(`(() => {
      const rect = document.querySelector('.card-btn').getBoundingClientRect()
      return { width: rect.width, height: rect.height }
    })()`)
    await fetch(`http://127.0.0.1:${debugPort}/json/activate/${gameTarget.id}`)
    // Chrome 151 leaves captureVisibleTab pending in --headless=new. Supply a
    // real browser-rendered PNG at that single API boundary; the extension's
    // crop, resampling, WASM inference, highlighting, and action path are real.
    const screenshot = await game.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: false,
      fromSurface: true,
    })
    await game.send('Page.navigate', {
      url: `http://127.0.0.1:${previewPort}/?vision-board=${testCase.theme}&board-seed=${1100 + index}&vision-seed=${19000 + index}`,
    })
    await waitFor(
      () => game.evaluate("document.documentElement.dataset.visionBoardReady === 'true'"),
      'alternate responsive board',
    )
    const changedScreenshot = await game.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: false,
      fromSurface: true,
    })
    await game.send('Page.navigate', {
      url: `http://127.0.0.1:${previewPort}/?vision-board=${testCase.theme}&board-seed=${100 + index}&vision-seed=${9000 + index}`,
    })
    await waitFor(
      () => game.evaluate("document.documentElement.dataset.visionBoardReady === 'true'"),
      'restored responsive board',
    )
    const gameTab = await panel.evaluate(
      `chrome.tabs.query({}).then((tabs) => tabs.find((tab) => tab.url?.startsWith(${JSON.stringify(`http://127.0.0.1:${previewPort}/`)})))`,
    )
    if (!gameTab?.id) throw new Error('Could not resolve the responsive-board tab')
    await panel.evaluate(
      `globalThis.__setVisionCaptureCount = 0
      globalThis.__setVisionHeadlessIo = {
        captureVisibleTab: async function () {
          const value = globalThis.__setVisionCaptureCount++ < 3
            ? ${JSON.stringify(`data:image/png;base64,${screenshot.data}`)}
            : ${JSON.stringify(`data:image/png;base64,${changedScreenshot.data}`)}
          return value
        },
        activeTab: async () => (${JSON.stringify(gameTab)}),
      }`,
    )
    await fetch(`http://127.0.0.1:${debugPort}/json/activate/${panelTarget.id}`)
    const screenshotProbe = await panel.evaluate(`Promise.race([
      (async () => {
        const dataUrl = await globalThis.__setVisionHeadlessIo.captureVisibleTab()
        const image = new Image()
        image.src = dataUrl
        await image.decode()
        return { ok: true, width: image.naturalWidth, height: image.naturalHeight }
      })(),
      new Promise((resolve) => setTimeout(() => resolve({ ok: false, timeout: true }), 2_000)),
    ])`)
    if (!screenshotProbe?.ok) {
      throw new Error(`Headless screenshot adapter failed: ${JSON.stringify(screenshotProbe)}`)
    }
    await panel.evaluate("document.querySelector('#status').textContent = 'test pending'; document.querySelector('#scan').click()")
    const status = await waitFor(async () => {
      const value = await panel.evaluate("document.querySelector('#status').textContent")
      if (value.startsWith('Scored') || value.startsWith('Abstained:') || value.startsWith('No SET predicted;')) return value
      if (
        value !== 'test pending'
        && !value.startsWith('Capturing ')
        && !value.startsWith('Running local policy ')
      ) throw new Error(value)
      if (value !== 'test pending') throw new Error(`still ${value}`)
      return undefined
    }, `${testCase.theme} ${width}x${height} scan`, 20_000)
    const positions = await panel.evaluate(
      "document.querySelector('#choice').textContent.split('·').map(value => Number(value.trim()) - 1)",
    )
    const ids = await game.evaluate(
      `(${JSON.stringify(positions)}).map(index => Number(document.querySelectorAll('.card-btn')[index].dataset.cardId))`,
    )
    const valid = isSet(ids)
    modelFailures += Number(!valid)
    const playEnabled = await panel.evaluate("!document.querySelector('#play').disabled")
    acceptedActions += Number(playEnabled)
    if (playEnabled) await panel.evaluate("document.querySelector('#play').click(); true")
    const clicked = playEnabled ? await waitFor(async () => {
      const selected = await game.evaluate(`Array.from(document.querySelectorAll('.card-btn'))
        .map((card, index) => card.getAttribute('aria-pressed') === 'true' ? index : -1)
        .filter((index) => index >= 0)`)
      return selected.length === 3 ? selected : undefined
    }, `${testCase.theme} ${width}x${height} click action`) : []
    const actionValid = !playEnabled || JSON.stringify(clicked) === JSON.stringify(positions)
    const transitionVerified = playEnabled ? await waitFor(async () => {
      const value = await panel.evaluate("document.querySelector('#status').textContent")
      if (value === 'Board change verified from pixels') return true
      if (value.startsWith('Auto-play stopped:') || value.includes('did not change')) throw new Error(value)
      return undefined
    }, `${testCase.theme} ${width}x${height} board transition`, 20_000) : false
    actionFailures += Number(playEnabled && (!actionValid || !transitionVerified))
    const casePassed = valid && actionValid && (!playEnabled || transitionVerified)
    console.log(`${casePassed ? 'PASS' : 'FAIL'} ${testCase.theme.padEnd(9)} viewport=${width}x${height} card=${cardRect.width.toFixed(1)}x${cardRect.height.toFixed(1)} ids=${ids.join(',')} ${status}`)
  }
  game.close()
  panel.close()
  if (modelFailures > 0 || actionFailures > 0 || acceptedActions === 0) {
    throw new Error(
      `${modelFailures}/${cases.length} invalid selections; ${actionFailures}/${acceptedActions} accepted action mismatches; ${acceptedActions}/${cases.length} actions accepted`,
    )
  }
  console.log(`extension screenshot path: ${cases.length}/${cases.length} valid proposals; ${acceptedActions}/${cases.length} guarded actions clicked and transition-verified`)
} finally {
  for (const process of processes.reverse()) process.kill('SIGTERM')
  if (temporaryRoot) await rm(temporaryRoot, { recursive: true, force: true })
}
